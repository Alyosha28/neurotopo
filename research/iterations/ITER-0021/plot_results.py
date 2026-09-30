"""Frozen reporting plots, no model execution or outcome selection."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
run=args.run;out=args.output;out.mkdir(parents=True,exist_ok=True)
c=json.loads((run/'completion.json').read_text());a=json.loads((run/'analysis.json').read_text());g=json.loads((run/'geometry_summary.json').read_text())
lookup={(r['seed'],r['arm'],r['cell'],r['rotation'],r['condition']):r for r in c['rows'] if r['phase']=='main'}
ks=[2,4,8,16,32,64];colors={25002:'#0072B2',25003:'#D55E00'}
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
fig,axes=plt.subplots(2,2,figsize=(11,7),sharex=True,sharey=True)
for ri,rotation in enumerate(('ORIGINAL','IDENTITY')):
 for ci,cell in enumerate(('LONG_H','LONG_U')):
  ax=axes[ri,ci]
  for seed in range(25001,25013):
   y=[lookup[seed,'REL',cell,rotation,'K'+str(k)]['summary']['metrics']['final_1024'] for k in ks]
   ax.plot(ks,y,'o-',color=colors.get(seed,'#999999'),lw=2 if seed in colors else .7,alpha=1 if seed in colors else .45,label=str(seed) if seed in colors else None)
  ax.axhline(.95,color='#444444',ls=':',lw=1);ax.set_xscale('log',base=2);ax.set_xticks(ks,labels=ks);ax.set_ylim(-.02,1.02)
  ax.set_title(f'REL {cell}: {rotation} rotations');ax.set_xlabel('Refresh interval k');ax.set_ylabel('Terminal accuracy at T=1024')
  ax.legend(loc='lower left')
fig.suptitle('All 12 exposed REL seeds; highlighted cases were fixed before this run')
fig.tight_layout();fig.savefig(out/'accuracy_vs_interval.png',dpi=170);fig.savefig(out/'accuracy_vs_interval.svg');plt.close(fig)

fig,axes=plt.subplots(2,3,figsize=(15,7))
conds=('NONE','ONCE8','K2','K4','K8','K16','K32','K64');palette=plt.get_cmap('tab10')
for i,seed in enumerate((25002,25003)):
 for j,cond in enumerate(conds):
  row=lookup[seed,'REL','LONG_H','ORIGINAL',cond]
  with np.load(run/row['raw_path']) as f,np.load(row['input_path']) as data:
   d=f['distance_pre'].mean(0);acc=((f['logits']>=0)==data['targets']).mean(0)
  axes[i,0].plot(np.arange(len(d)),np.maximum(d,1e-12),label=cond,color=palette(j),lw=1)
  axes[i,1].plot(np.arange(81),acc[:81],label=cond,color=palette(j),lw=1.5)
  axes[i,2].plot(np.arange(len(acc)),acc,label=cond,color=palette(j),lw=1.5)
 axes[i,0].axhline(row['theta'],ls=':',color='black',label='calibration q90')
 axes[i,0].set_yscale('log');axes[i,0].set_ylabel(f'{seed}: mean Gram distance')
 axes[i,1].set_ylabel(f'{seed}: HOLD accuracy');axes[i,1].set_ylim(-.02,1.02);axes[i,2].set_ylim(-.02,1.02)
 axes[i,0].set_title('Distance, full trajectory');axes[i,1].set_title('Accuracy, first 80 steps');axes[i,2].set_title('Accuracy, full trajectory')
 for ax in axes[i]:ax.set_xlabel('Step')
axes[0,2].legend(fontsize=8,ncol=2,loc='lower right')
fig.suptitle('Original rotations; self-selected prototype distance (display floor 1e-12)')
fig.tight_layout();fig.savefig(out/'distance_and_accuracy.png',dpi=170);fig.savefig(out/'distance_and_accuracy.svg');plt.close(fig)

fig,ax=plt.subplots(figsize=(9,5));not_available=[]
windows={r['seed']:r for r in a['seed_windows'] if r['rotation']=='ORIGINAL'}
for row in g['command_reset_geometry']:
 seed,arm,cell,rotation,valid,total,mean,median,_,_,_=row
 if arm!='REL' or rotation!='ORIGINAL' or seed==23005:continue
 if mean is None:not_available.append(f'{seed}/{cell}');continue
 y=sum(windows[seed]['k_success'])/6
 ax.scatter(mean,y,marker='o' if cell=='LONG_H' else 's',s=35,color=colors.get(seed,'#777777'),alpha=.75)
 if seed in colors:
  offset=((10,46) if cell=='LONG_H' else (-150,18)) if seed==25003 else (3,4)
  ax.annotate(f'{seed} {cell[-1]} ({valid}/{total})',(mean,y),xytext=offset,textcoords='offset points',fontsize=8,arrowprops={'arrowstyle':'-','lw':.5} if seed==25003 else None)
ax.set_xlabel('Mean cosine: paired command response vs removed Gram component')
ax.set_ylabel('Fraction of six k passing the fixed success conjunction')
ax.set_ylim(-.08,1.08);ax.set_xlim(-1.05,1.05);ax.set_title('Descriptive association: circles=HOLD, squares=UPDATE; other seeds grey')
fig.text(.02,.01,'Unavailable overlap values: '+(', '.join(not_available) if not_available else 'none'),fontsize=7)
fig.tight_layout(rect=(0,.04,1,1));fig.savefig(out/'command_overlap_vs_windows.png',dpi=170);fig.savefig(out/'command_overlap_vs_windows.svg');plt.close(fig)
for svg in out.glob('*.svg'):
 if not svg.name.startswith('._'):
  svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
print('three figures written as PNG + SVG')
