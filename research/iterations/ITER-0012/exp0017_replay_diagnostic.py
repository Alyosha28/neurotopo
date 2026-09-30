import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from neurotopo.order_observation import generate
from neurotopo.order_observation_reproduce import independent_kl, log_oracle
from neurotopo.run import source_fingerprints

run = Path('artifacts/EXP-0017/v2')
out = Path('research/iterations/ITER-0012/reproduction.json')
started = time.monotonic()
torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)
manifest = json.loads((run / 'manifest.json').read_text())
analysis = json.loads((run / 'analysis.json').read_text())
assert manifest['status'] == 'COMPLETED'
assert manifest['source_sha256'] == source_fingerprints()
assert manifest['git_commit'] == 'fe0907ba99b63b7aa76cf218ded5d43ea7454fc3'
for name, expected in manifest['frozen_hashes'].items():
    assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == expected
assert len(manifest['records']) == 24 and manifest['research_updates'] == 18000
records = []
max_q = max_z = max_kl = 0.0
paired = {}
for ci, cell in enumerate(('CO', 'CU', 'NO', 'NU')):
    for seed in (17001, 17002, 17003):
        for arm in ('D16', 'D64'):
            path = run / f'seed-{seed}' / cell / arm
            stored = json.loads((path / 'result.json').read_text())
            assert stored['status'] == 'COMPLETED'
            assert stored['updates'] == len(stored['losses']) == 750
            for name, key in (('final.pt', 'checkpoint_sha256'), ('predictions.npz', 'predictions_sha256')):
                assert hashlib.sha256((path / name).read_bytes()).hexdigest() == stored[key]
            paired.setdefault((cell, seed), []).append(stored['training_data_sha256'])
            h = int(arm[1:])
            state = torch.load(path / 'final.pt', map_location='cpu', weights_only=True)['model']
            gru = nn.GRU(5, h, batch_first=True)
            readout = nn.Linear(h, 6)
            gru.load_state_dict({k.removeprefix('gru.'): v for k, v in state.items() if k.startswith('gru.')})
            readout.load_state_dict({k.removeprefix('readout.'): v for k, v in state.items() if k.startswith('readout.')})
            scores = {}
            with np.load(path / 'predictions.npz') as arrays:
                for n in (32, 128):
                    data = generate(4096, n, cell, [172001, seed, ci, n])
                    tokens = arrays[f'tokens_{n}']
                    assert np.array_equal(tokens, data['tokens'])
                    assert np.array_equal(arrays[f'latent_{n}'], data['latent_final'])
                    q = log_oracle(tokens, cell)[:, -1]
                    eq = float(np.max(np.abs(q - arrays[f'target_{n}'])))
                    assert eq <= 1e-12
                    logits = []
                    with torch.no_grad():
                        for i in range(0, 4096, 128):
                            x = F.one_hot(torch.from_numpy(tokens[i:i+128].astype(np.int64)), 5).float()
                            hidden, _ = gru(x)
                            logits.append(readout(hidden)[:, -1].numpy())
                    z = np.concatenate(logits)
                    ez = float(np.max(np.abs(z - arrays[f'logits_{n}'])))
                    kl = independent_kl(q, z)
                    ek = float(np.max(np.abs(kl - arrays[f'kl_{n}'])))
                    assert ez <= 1e-6 and ek <= 1e-6
                    assert abs(float(kl.mean()) - stored['metrics'][str(n)]['kl_mean']) <= 1e-6
                    max_q, max_z, max_kl = max(max_q, eq), max(max_z, ez), max(max_kl, ek)
                    scores[str(n)] = float(kl.mean())
            records.append(dict(cell=cell, seed=seed, arm=arm, kl=scores))
assert all(len(x) == 2 and x[0] == x[1] for x in paired.values())
assert {(x['cell'], x['seed'], x['arm']) for x in manifest['records']} == {(x['cell'], x['seed'], x['arm']) for x in records}
learning, headroom, cell_stats = {}, {}, {}
for cell in ('CO', 'CU', 'NO', 'NU'):
    rows = [x for x in records if x['cell'] == cell and x['arm'] == 'D64']
    learning[cell] = all(x['kl']['32'] < .1 for x in rows)
    headroom[cell] = bool(np.mean([x['kl']['128'] for x in rows]) >= .02)
    cell_stats[cell] = {}
    for n, label in (('32', 'iid'), ('128', 'ood')):
        values = [x['kl'][n] for x in rows]
        mean = float(np.mean(values))
        sd = float(np.std(values, ddof=1))
        ci = [mean - 4.302652729911275 * sd / math.sqrt(3), mean + 4.302652729911275 * sd / math.sqrt(3)]
        assert abs(mean - analysis['cells'][cell][label]['mean']) <= 1e-6
        assert np.max(np.abs(np.asarray(ci) - analysis['cells'][cell][label]['t95_df2'])) <= 1e-6
        cell_stats[cell][label] = dict(values=values, mean=mean, sample_sd=sd, t95_df2=ci)
    assert learning[cell] == analysis['cells'][cell]['learning_pass']
    assert headroom[cell] == analysis['cells'][cell]['headroom_pass']
substatus = 'LEARNING_FAILURE' if not all(learning.values()) else 'CONTROL_CEILING' if not all(headroom.values()) else 'CONTROL_PILOT_WINDOW'
decision = 'SUPPORTED' if substatus == 'CONTROL_PILOT_WINDOW' else 'INCONCLUSIVE'
assert (decision, substatus) == (analysis['decision'], analysis['substatus'])
assert time.monotonic() - started <= 300
report = dict(
    frozen_module_pass=False,
    frozen_module_failure=dict(cell='CO', seed=17001, arm='D16', stage='logit/KL tolerance assertion',
                               error='AssertionError at order_observation_reproduce.py:95',
                               max_logit_error_32=1.9073486328125e-6, max_kl_error_32=9.613802782748593e-7,
                               reason='readout(hidden[:, -1]) uses a different floating-point matrix shape from the registered readout(hidden)[:, -1]'),
    diagnostic_pass=True,
    diagnostic_scope='24 frozen checkpoints, both evaluation lengths, independent log-domain oracle and KL, exact registered forward order; no retraining',
    decision=decision, substatus=substatus, records=records, learning=learning, headroom=headroom,
    cells=cell_stats, max_posterior_error=max_q, max_logit_error=max_z, max_kl_error=max_kl,
    seconds=time.monotonic()-started, source_sha256=source_fingerprints(),
    run_manifest_sha256=hashlib.sha256((run/'manifest.json').read_bytes()).hexdigest(),
    replay_program_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
)
with out.open('x') as stream:
    json.dump(report, stream, indent=2, allow_nan=False)
    stream.write('\n')
print(json.dumps(dict(diagnostic_pass=True, decision=decision, substatus=substatus,
                      max_posterior_error=max_q, max_logit_error=max_z, max_kl_error=max_kl,
                      seconds=report['seconds'], learning=learning, headroom=headroom)))
