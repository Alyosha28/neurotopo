import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--report-dir',type=Path,required=True);a=p.parse_args()
r=json.loads((a.run/'results.json').read_text());key=json.loads((a.report_dir/'key_findings.json').read_text());rates=json.loads((a.report_dir/'strict_rates.json').read_text())['rows'];out=a.report_dir/'figures';out.mkdir(exist_ok=True)
plt.rcParams.update({'font.size':10,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(1,2,figsize=(11,4.8));colors=['#666666','#0072B2','#009E73','#D55E00']
for ax,kind,title in zip(axes,('controlled','natural'),('Controlled suffix: one FLIP','Original UPDATE suffix: four FLIPs')):
 rows=key[kind]['comparison'];values=[x['future_logit'] for x in rows];ax.bar(range(4),values,color=colors);ax.axhline(0,color='black',lw=.8);ax.set_xticks(range(4),[x['method'] for x in rows]);ax.set_yscale('symlog',linthresh=.1);ax.set_ylim(-.25,6)
 for i,v in enumerate(values):ax.annotate(f'{v:+.6f}',(i,v),xytext=(0,6 if v>=0 else -13),textcoords='offset points',ha='center',fontsize=9)
 z=key[kind]['witness']['event_pre_logit'];ax.set_title(f'{title}\nBefore / after EQUAL logit: {z:.6f}');ax.set_ylabel('Future logit (symlog; true class = 1)')
fig.suptitle('Same current readout, different future computation: REL 25003');fig.tight_layout();fig.savefig(out/'short_counterexamples.png',dpi=170);fig.savefig(out/'short_counterexamples.svg');plt.close(fig)

models=[tuple(x) for x in r['models']];table=[];texts=[]
for seed,arm in models:
 bs=[x for x in r['blocks'] if x['seed']==seed and x['arm']==arm and x['method']=='FULL'];bs.sort(key=lambda x:x['k']);w=bs[0]['warmup']
 vals=[bs[0]['all_time_correct_cases']/8,bs[1]['all_time_correct_cases']/32,w['all_time_correct']/512,w['correct_canonical_entry']/512];table.append(vals)
 texts.append([f"{bs[0]['all_time_correct_cases']}/8",f"{bs[1]['all_time_correct_cases']}/32",f"{w['all_time_correct']}/512",f"{w['correct_canonical_entry']}/512"])
fig,ax=plt.subplots(figsize=(10,5.5));ax.imshow(table,vmin=.7,vmax=1,cmap='RdYlGn',aspect='auto');ax.set_xticks(range(4),['FULL 2-step block\nall scored times','FULL 4-step block\nall scored times','8-step warmup\nall scored times','Warmup canonical\nentry at step 8']);ax.set_yticks(range(8),[f'{s} {arm}' for s,arm in models])
for i in range(8):
 for j in range(4):ax.text(j,i,texts[i][j],ha='center',va='center',color='white' if table[i][j]>.97 else 'black')
ax.set_title('Finite FULL reset checks: closure succeeds; initialization can fail')
fig.text(.02,.01,'Identity rotations only. Warmup includes command orders absent from the original training mixture.',fontsize=8);fig.tight_layout(rect=(0,.04,1,1));fig.savefig(out/'block_and_initialization.png',dpi=170);fig.savefig(out/'block_and_initialization.svg');plt.close(fig)

branches=['EQUAL_before','EQUAL_after1','EQUAL_after2','FULL_before','FULL_after1','FULL_after2'];values=np.zeros((8,6));counts={}
for i,(seed,arm) in enumerate(models):
 for j,b in enumerate(branches):
  rs=[x for x in rates if x['group']=='natural' and x['seed']==seed and x['arm']==arm and x['branch']==b];n=sum(x['strict_eligible'] for x in rs);bad=sum(x['damage'] for x in rs);values[i,j]=100*bad/n if n else np.nan;counts[i,j]=(bad,n)
fig,ax=plt.subplots(figsize=(13,6));im=ax.imshow(values,aspect='auto',vmin=0,cmap='YlOrRd');ax.set_xticks(range(6),[x.replace('_','\n') for x in branches]);ax.set_yticks(range(8),[f'{s} {arm}' for s,arm in models]);ax.set_title('Original UPDATE: conditional harm in the same 32-step suffix')
for i in range(8):
 for j in range(6):
  bad,n=counts[i,j];ax.text(j,i,f'{bad}/{n}\n{values[i,j]:.2f}%' if n else 'unavailable',ha='center',va='center',fontsize=8,color='white' if np.isfinite(values[i,j]) and values[i,j]>3 else 'black')
fig.colorbar(im,ax=ax,label='Harm percentage among strict eligible tests');fig.text(.02,.01,'Overlapping snapshots are not independent models. Eligible: baseline correct, valid reset, immediate class preserved, GAUGE control and margin checks pass.',fontsize=8)
fig.tight_layout(rect=(0,.04,1,1));fig.savefig(out/'natural_update_harm.png',dpi=170);fig.savefig(out/'natural_update_harm.svg');plt.close(fig)
for f in out.glob('*.svg'):
 if not f.name.startswith('._'):f.write_text('\n'.join(s.rstrip() for s in f.read_text().splitlines())+'\n')
print('3 PNG and 3 SVG figures saved')
