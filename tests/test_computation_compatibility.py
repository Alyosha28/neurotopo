import numpy as np
import torch
from torch import nn
from neurotopo.computation_compatibility import branch,block_test,paired,periodic,targets
from neurotopo.learned_protected_pilot import LearnedMemory,generate,gram
from neurotopo.state_refresh_core import fit_prototypes
from neurotopo.schedule_core import rollout,threshold

class SwapMemory(nn.Module):
    def __init__(self):
        super().__init__();self.readout=nn.Linear(3,1)
        with torch.no_grad():self.readout.weight.copy_(torch.tensor([[1.,0.,-1.]]));self.readout.bias.zero_()
    def transition(self,h,u):return torch.where(u[:,None,None].bool(),h.flip(-1),h)

def prototypes():
    p=np.zeros((2,3,2),np.float32);p[0,0,1]=1;p[1,0,0]=1;return p

def test_full_finite_block_certificate_and_equal_closed_example():
    model=SwapMemory();p=prototypes()
    for k in (2,4):
        for method in ('FULL','EQUAL'):
            summary,raw=block_test(model,p,k,method)
            assert summary['unique_cases']==2*2**k and summary['all_correct_and_closed']
            assert np.all(raw['canonical_return'])

def test_correct_swap_model_has_no_spurious_damage_and_fixed_scoring():
    p=prototypes();h=np.repeat(p,64,axis=0);labels=np.repeat([0,1],64);u=np.tile([1,0,1,1],(128,1))
    summary,raw=paired(SwapMemory(),h,labels,u,p,{'kind':'test'})
    assert all(row['damage_tests']==0 for row in summary['branches'])
    expected=targets(labels,u)
    for key,val in raw.items():
        if key.endswith('__pre_logits'):assert np.array_equal(val>=0,expected)

class WritableMemory(nn.Module):
    def __init__(self):
        super().__init__();self.readout=nn.Linear(3,1)
        with torch.no_grad():self.readout.weight.copy_(torch.tensor([[1.,0.,0.]]));self.readout.bias.fill_(-1)
    def transition(self,h,u):
        g=gram(h);a=g[:,0];write=(u==1)&(g[:,2]>.5)
        a=torch.where(write,2-a,a).clamp_min(0)
        out=h.clone();out[:,:,0]=0;out[:,0,0]=torch.sqrt(a);return out

def test_equal_preserves_now_but_can_destroy_next_update():
    p=np.zeros((2,3,2),np.float32);p[1,0,0]=np.sqrt(2.)
    h=np.repeat(p[1:2],128,axis=0);h[:,1,1]=1
    u=np.tile([1,0,0,0],(128,1));labels=np.ones(128,np.int64)
    summary,raw=paired(WritableMemory(),h,labels,u,p,{'kind':'known_counterexample'})
    r=next(row for row in summary['branches'] if row['branch']=='EQUAL_before')
    assert r['damage_tests']==128 and r['gauge_confounded_damage_tests']==0
    w=next(w for w in summary['witnesses'] if w['branch']=='EQUAL_before')
    assert w['steps_after_event']==1 and w['failing_suffix']==[1]
    assert abs(w['event_pre_logit']-w['event_post_logit'])<=1e-5

def test_existing_periodic_phase_replays_registered_schedule():
    torch.set_num_threads(1);m=LearnedMemory('REL',279027)
    cal=generate(128,8,[279027,1],'MIX');data=generate(128,32,[279027,2],'UPDATE',np.pi,0.);p,_=fit_prototypes(m,cal)
    for k,phase,old in ((4,0,'K4'),(16,0,'PHASE16'),(16,8,'K16')):
        a=periodic(m,data,p['prototypes'],k,phase)
        b=rollout(m,data,p['prototypes'],old,'ORIGINAL',threshold(p))
        np.testing.assert_array_equal(a['logits'],b['logits'])

def test_snapshot_extraction_preserves_original_batch_semantics():
    from neurotopo.computation_compatibility import snapshots,snapshot_readout,ANCHORS
    from neurotopo.learned_protected_pilot import predict_with_gram
    torch.set_num_threads(1);m=LearnedMemory('REL',279027)
    data=generate(512,32,[279027,3],'UPDATE',np.pi,0.)
    z,g=predict_with_gram(m,data);hs=snapshots(m,data)
    for t in ANCHORS:
        actual_g,actual_z=snapshot_readout(m,hs[t])
        np.testing.assert_array_equal(actual_g,g[:,t]);np.testing.assert_array_equal(actual_z,z[:,t])
