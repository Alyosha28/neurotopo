"""Minimal full-state reproducer; needs only the frozen checkpoint + key_findings.json."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import torch
from neurotopo.learned_protected_pilot import LearnedMemory,gram
p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--case',choices=['controlled','natural'],default='controlled');a=p.parse_args()
k=json.loads(Path(__file__).with_name('key_findings.json').read_text())[a.case];w=k['witness']
assert hashlib.sha256(a.checkpoint.read_bytes()).hexdigest()==w['checkpoint_sha256']
torch.set_num_threads(1);m=LearnedMemory(w['arm'],w['seed']);m.load_state_dict(torch.load(a.checkpoint,map_location='cpu',weights_only=True)['model']);m.eval()
word=np.array(w['failing_suffix'],np.int64);expected=np.bitwise_xor.accumulate(np.r_[w['label_at_event'],word]).tolist();trajectories={}
with torch.no_grad():
 for name,field in [('NONE','full_state_before'),('EQUAL','full_state_after')]:
  h=torch.tensor(w[field],dtype=torch.float32).unsqueeze(0).repeat(128,1,1);z=[float(m.readout(gram(h))[0,0])]
  for t,u in enumerate(word):
   R=torch.tensor(w['rotation_suffix'][t],dtype=torch.float32).expand(128,3,3) if 'rotation_suffix' in w else torch.eye(3).expand(128,3,3)
   E=torch.tensor(w['leak_suffix'][t],dtype=torch.float32).expand(128,3,3) if 'leak_suffix' in w else torch.eye(3).expand(128,3,3)
   h=R@h;h=m.transition(torch.stack((h[:,:,0],(E@h[:,:,1:2]).squeeze(-1)),dim=-1),torch.full((128,),int(u),dtype=torch.int64));z.append(float(m.readout(gram(h))[0,0]))
  trajectories[name]={'logits':z,'predictions':[int(v>=0) for v in z]}
assert trajectories['NONE']['predictions']==expected
assert trajectories['EQUAL']['predictions'][0]==expected[0] and trajectories['EQUAL']['predictions'][-1]!=expected[-1]
assert abs(trajectories['NONE']['logits'][0]-trajectories['EQUAL']['logits'][0])<=1e-5
print(json.dumps({'case':a.case,'seed':w['seed'],'commands':word.tolist(),'expected':expected,'branches':trajectories,'verified':True},indent=2))
