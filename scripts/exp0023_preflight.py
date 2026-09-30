"""Declared engineering fit/profile only: root239023, no research checkpoint."""
from pathlib import Path
import time
import torch
from neurotopo.learned_protected_pilot import LearnedMemory,generate
from neurotopo.composition_core import loss_terms
from neurotopo.run import source_fingerprints,write_json
from neurotopo.order_observation import sha

def main():
    output=Path('artifacts/validation/exp0023_preflight.json')
    if output.exists():raise FileExistsError('Preserve receipt; special fit is single invocation')
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    fit=generate(16,1,[239023,2],'MIX');rows=[];updates=0
    for variant in ('REL_BASE','REL_CYCLE','REL_EXTRA'):
        m=LearnedMemory('REL',239023);opt=torch.optim.Adam(m.parameters(),lr=.003)
        initial=float(loss_terms(m,fit,variant)[0].detach())
        for step in range(30):
            opt.zero_grad(set_to_none=True);total,_,_=loss_terms(m,fit,variant);total.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(),1.,error_if_nonfinite=True);opt.step();updates+=1
        final=float(loss_terms(m,fit,variant)[0].detach())
        rows.append({'variant':variant,'initial_total_loss':initial,'final_total_loss':final,'updates':30,'pass':final<initial-1e-4})
    data=generate(64,8,[239023,3],'MIX');profiles=[]
    for variant in ('REL_CYCLE','REL_EXTRA'):
        m=LearnedMemory('REL',239023);start=time.perf_counter()
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU],record_shapes=True,with_flops=True) as prof:
            total,_,_=loss_terms(m,data,variant);total.backward()
        events={e.key:{'count':e.count,'profiled_flops':e.flops,'self_cpu_time_us':e.self_cpu_time_total} for e in prof.key_averages()}
        dense={k:v for k,v in events.items() if k in ('aten::mm','aten::addmm','aten::bmm')}
        profiles.append({'variant':variant,'seconds_with_profiler':time.perf_counter()-start,'dense_ops':dense,'all_operator_counts':{k:v['count'] for k,v in events.items()},'profiled_dense_matmul_flops':sum(v['profiled_flops'] for v in dense.values()),'scope':'One untrained batch forward+backward; profiler-supported dense matmul only, not total training FLOPs; no optimizer update.'})
    a,b=profiles
    matched=a['profiled_dense_matmul_flops']==b['profiled_dense_matmul_flops'] and {k:v['count'] for k,v in a['dense_ops'].items()}=={k:v['count'] for k,v in b['dense_ops'].items()}
    passed=all(r['pass'] for r in rows) and matched
    write_json(output,{'pass':passed,'source_sha256':source_fingerprints(),'config_sha256':sha(Path('configs/exp0023_v2.json')),'special_fit_updates':updates,'fit':rows,'operator_profiles':profiles,'dense_operations_matched':matched,'scope':'Engineering seeds only; no outcome-based coefficient or candidate selection.'})
    if not passed:raise RuntimeError('Engineering preflight failed; receipt preserved')
if __name__=='__main__':main()
