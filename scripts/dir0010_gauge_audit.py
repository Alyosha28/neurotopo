"""Read-only numeric precision audit for leak-zero common rotations."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from neurotopo.order_observation import sha
from neurotopo.run import source_fingerprints

SEEDS=tuple(range(25001,25013))+(23005,)
CELLS=('LONG_H','LONG_U')
IDENTITY=np.eye(3)


def audit(root:Path)->dict:
    records=[]
    for seed in SEEDS:
        for cell in CELLS:
            path=root/'data'/str(seed)/f'{cell}.npz'
            with np.load(path) as f:
                r=f['common'];leak=f['leak'];commands=f['commands'];target=f['targets'];b0=f['b0']
            if r.shape!=(512,1025,3,3) or leak.shape!=(512,1024,3,3) or commands.shape!=(512,1024) or target.shape!=(512,1025):
                raise RuntimeError('Frozen matrix shape mismatch')
            oracle=np.bitwise_xor.accumulate(np.column_stack((b0,commands)),axis=1)
            if not np.array_equal(oracle,target):raise RuntimeError('Task target mismatch')
            if np.max(np.abs(leak.astype(np.float64)-IDENTITY))!=0:raise RuntimeError('Leak is not identity')
            # Input matrices are stored float32. The operation below is float64
            # and measures the float32 matrix's deviation from exact orthogonality.
            rr=r.astype(np.float64)
            residual=np.einsum('...ki,...kj->...ij',rr,rr)-IDENTITY
            norm=np.linalg.norm(residual,axis=(-2,-1))
            det=np.linalg.det(rr)
            if not np.isfinite(norm).all() or not np.isfinite(det).all():raise FloatingPointError('Nonfinite rotation audit')
            records.append({'seed':seed,'cell':cell,'input_sha256':sha(path),
                'matrices':int(norm.size),'orthogonality_fro_mean':float(norm.mean()),
                'orthogonality_fro_median':float(np.median(norm)),
                'orthogonality_fro_p99':float(np.quantile(norm,.99)),
                'orthogonality_fro_max':float(norm.max()),
                'fraction_fro_above_1e_minus_6':float(np.mean(norm>1e-6)),
                'determinant_abs_error_mean':float(np.mean(np.abs(det-1.))),
                'determinant_abs_error_max':float(np.max(np.abs(det-1.))),
                'leak_identity_max_error':0.,'oracle_exact':True})
    result={'task':'DIR-0010','scope':'read-only float64 audit of saved float32 SO(3) matrices; no model forward',
            'source_sha256':source_fingerprints(),'input_root':str(root.resolve()),
            'input_count':len(records),'total_rotation_matrices':sum(x['matrices'] for x in records),
            'global_max_orthogonality_fro':max(x['orthogonality_fro_max'] for x in records),
            'global_max_determinant_abs_error':max(x['determinant_abs_error_max'] for x in records),
            'records':records}
    return result

if __name__=='__main__':
    result=audit(Path('artifacts/DYN-0001/v3'))
    out=Path('research/evidence/DIR-0010_rotation_precision.json')
    with out.open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps({k:result[k] for k in ('task','input_count','total_rotation_matrices','global_max_orthogonality_fro')}))
