"""Independent log-domain oracle and frozen-checkpoint replay; no training."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .run import write_json, source_fingerprints


def log_oracle(tokens: np.ndarray, cell: str) -> np.ndarray:
    r = [(q // 3) * 3 + (q % 3 + 1) % 3 for q in range(6)]
    s = [(1 - q // 3) * 3 + (q % 3 if cell[0] == 'C' else (-q % 3)) for q in range(6)]
    state = np.full((len(tokens), 6), -np.inf)
    state[:, 0] = 0.
    history = []
    for token in tokens.T:
        following = np.full_like(state, -np.inf)
        for source in range(6):
            for a, mapping in enumerate((r, s)):
                contribution = state[:, source] + np.where(token == a, 0., np.where(token == 2, -np.log(2.), -np.inf))
                following[:, mapping[source]] = np.logaddexp(following[:, mapping[source]], contribution)
            observed = token >= 3
            compatible = (source % 3 == 0) == (token == 4)
            following[:, source] = np.where(observed, np.where(compatible, state[:, source], -np.inf), following[:, source])
        norm = np.logaddexp.reduce(following, axis=1, keepdims=True)
        if not np.isfinite(norm).all():
            raise ValueError('Impossible observation')
        state = following - norm
        history.append(np.exp(state))
    return np.stack(history, axis=1)


def independent_kl(q: np.ndarray, logits: np.ndarray) -> np.ndarray:
    logp = logits.astype(np.float64) - np.logaddexp.reduce(logits.astype(np.float64), axis=1)[:, None]
    entropy = np.zeros_like(q)
    active = q > 0
    entropy[active] = q[active] * np.log(q[active])
    return entropy.sum(1) - (q * logp).sum(1)


def replay(run: Path, output: Path) -> dict:
    from .order_observation import generate
    if output.exists():
        raise FileExistsError(output)
    started = time.monotonic()
    manifest = json.loads((run / 'manifest.json').read_text())
    analysis = json.loads((run / 'analysis.json').read_text())
    assert manifest['status'] == 'COMPLETED' and manifest['source_sha256'] == source_fingerprints()
    for name, expected in manifest['frozen_hashes'].items():
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == expected
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    records, maximum_q, maximum_z, maximum_kl = [], 0., 0., 0.
    pairs = {}
    for cell_index, cell in enumerate(('CO', 'CU', 'NO', 'NU')):
        for seed in (17001, 17002, 17003):
            for arm in ('D16', 'D64'):
                if time.monotonic() - started > 300:
                    raise TimeoutError('Registered replay budget')
                path = run / f'seed-{seed}' / cell / arm
                stored = json.loads((path / 'result.json').read_text())
                assert stored['updates'] == len(stored['losses']) == 750
                for name, field in (('final.pt', 'checkpoint_sha256'), ('predictions.npz', 'predictions_sha256')):
                    assert hashlib.sha256((path / name).read_bytes()).hexdigest() == stored[field]
                pairs.setdefault((cell, seed), []).append(stored['training_data_sha256'])
                state = torch.load(path / 'final.pt', map_location='cpu', weights_only=True)['model']
                h = int(arm[1:])
                gru, readout = nn.GRU(5, h, batch_first=True), nn.Linear(h, 6)
                gru.load_state_dict({k.removeprefix('gru.'): v for k, v in state.items() if k.startswith('gru.')})
                readout.load_state_dict({k.removeprefix('readout.'): v for k, v in state.items() if k.startswith('readout.')})
                metrics = {}
                with np.load(path / 'predictions.npz') as arrays:
                    for n in (32, 128):
                        data = generate(4096, n, cell, [172001, seed, cell_index, n])
                        tokens = arrays[f'tokens_{n}']
                        assert np.array_equal(tokens, data['tokens']) and np.array_equal(arrays[f'latent_{n}'], data['latent_final'])
                        q = log_oracle(tokens, cell)[:, -1]
                        error_q = float(np.max(np.abs(q - arrays[f'target_{n}'])))
                        assert error_q <= 1e-12
                        logits = []
                        with torch.no_grad():
                            for i in range(0, 4096, 128):
                                x = F.one_hot(torch.from_numpy(tokens[i:i+128].astype(np.int64)), 5).float()
                                out, _ = gru(x)
                                # Match the registered forward's GEMM shape before slicing.
                                # Slicing first changes float32 rounding on this CPU backend.
                                logits.append(readout(out)[:, -1].numpy())
                        z = np.concatenate(logits)
                        error_z = float(np.max(np.abs(z - arrays[f'logits_{n}'])))
                        scores = independent_kl(q, z)
                        error_kl = float(np.max(np.abs(scores - arrays[f'kl_{n}'])))
                        assert error_z <= 1e-6 and error_kl <= 1e-6
                        assert abs(float(scores.mean()) - stored['metrics'][str(n)]['kl_mean']) <= 1e-6
                        maximum_q, maximum_z, maximum_kl = max(maximum_q,error_q), max(maximum_z,error_z), max(maximum_kl,error_kl)
                        metrics[str(n)] = float(scores.mean())
                records.append(dict(cell=cell, seed=seed, arm=arm, kl=metrics))
    assert all(len(x) == 2 and x[0] == x[1] for x in pairs.values())
    assert manifest['research_updates'] == 18000 and len(manifest['records']) == 24
    actual_keys = {(x['cell'], x['seed'], x['arm']) for x in manifest['records']}
    assert actual_keys == {(x['cell'], x['seed'], x['arm']) for x in records}
    learning, headroom = {}, {}
    for cell in ('CO', 'CU', 'NO', 'NU'):
        rows = [r for r in records if r['cell'] == cell and r['arm'] == 'D64']
        learning[cell] = all(r['kl']['32'] < .1 for r in rows)
        headroom[cell] = sum(r['kl']['128'] for r in rows) / 3 >= .02
        for n, key in (('32', 'iid'), ('128', 'ood')):
            vals = [r['kl'][n] for r in rows]
            mean, sd = np.mean(vals), np.std(vals, ddof=1)
            ci = [mean - 4.302652729911275 * sd / np.sqrt(3), mean + 4.302652729911275 * sd / np.sqrt(3)]
            assert abs(mean - analysis['cells'][cell][key]['mean']) <= 1e-6
            assert np.max(np.abs(np.asarray(ci) - analysis['cells'][cell][key]['t95_df2'])) <= 1e-6
    substatus = 'LEARNING_FAILURE' if not all(learning.values()) else 'CONTROL_CEILING' if not all(headroom.values()) else 'CONTROL_PILOT_WINDOW'
    decision = 'SUPPORTED' if substatus == 'CONTROL_PILOT_WINDOW' else 'INCONCLUSIVE'
    assert (decision, substatus) == (analysis['decision'], analysis['substatus'])
    result = dict(pass_=True, scope='Frozen-checkpoint replay and independent log-domain posterior; no retraining',
                  decision=decision, substatus=substatus, records=records, learning=learning, headroom=headroom,
                  max_posterior_error=maximum_q, max_logit_error=maximum_z, max_kl_error=maximum_kl,
                  seconds=time.monotonic()-started, source_sha256=source_fingerprints(), run_manifest_sha256=hashlib.sha256((run/'manifest.json').read_bytes()).hexdigest())
    assert result['seconds'] <= 300
    write_json(output, result)
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(replay(a.run, a.output)))
