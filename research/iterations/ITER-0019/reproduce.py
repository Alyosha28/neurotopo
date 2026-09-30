"""Independent EXP-0024 v1 replay. Run once against the immutable APFS output.

Model forward/transition and the registered generator come from the clean frozen
checkout on PYTHONPATH. Calibration algebra, intervention scheduling, scoring,
guards, statistics, verdict and provenance checks here are independent.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from neurotopo.learned_protected_pilot import LearnedMemory, generate, gram, radial, state_hash, tensors
from neurotopo.composition_core import predict_full
from neurotopo.protected_interventions import environment_step

ROOT = Path('/private/tmp/neurotopo-exp0024-review')
SEEDS = tuple(range(23001, 23007))
ARMS = ('REL_BASE', 'INV_BASE')
CONDS = ('NONE', 'GAUGE8', 'RESET8', 'RESET1')
CELLS = {'IID_H': (8, 'HOLD', .4, 1), 'IID_U': (8, 'UPDATE', .4, 2),
         'LONG_H': (256, 'HOLD', np.pi, 1), 'LONG_U': (256, 'UPDATE', np.pi, 2)}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def require(ok, message):
    if not ok:
        raise AssertionError(message)


def close(a, b, name, tol=1e-6):
    require(np.allclose(a, b, rtol=0, atol=tol, equal_nan=False), name)


def root_state(matrices):
    g = np.asarray(matrices, np.float64)
    g = (g + np.swapaxes(g, -1, -2)) / 2
    eig, vec = np.linalg.eigh(g)
    require(np.isfinite(eig).all() and eig.min() >= -1e-8, 'non PSD Gram')
    square = (vec * np.sqrt(np.maximum(eig, 0))[..., None, :]) @ np.swapaxes(vec, -1, -2)
    h = np.zeros((*square.shape[:-2], 3, 2), np.float32)
    h[..., :2, :] = square
    return h, eig


def model_at(path, seed, arm):
    state = torch.load(path, map_location='cpu', weights_only=True)
    require((state['seed'], state['variant'], state['arm'], state['step']) ==
            (seed, arm, 'REL' if arm == 'REL_BASE' else 'INV', 1000), 'checkpoint metadata')
    model = LearnedMemory('REL' if arm == 'REL_BASE' else 'INV', seed=seed)
    model.load_state_dict(state['model'])
    model.eval()
    return model


def independently_roll(model, data, condition, prototypes):
    count, length = data['commands'].shape
    mask = np.zeros(length + 1, bool)
    period = 1 if condition == 'RESET1' else 8
    if condition != 'NONE':
        mask[period::period] = True
    names = ('logits', 'gram', 'post_logits', 'post_gram', 'selected_labels')
    batches = {name: [] for name in names}
    proto = torch.from_numpy(prototypes)
    with torch.no_grad():
        for start in range(0, count, 128):
            b, u, r, e = tensors(data, start, start + 128)
            h = r[:, 0] @ radial(model.encoder(b).reshape(-1, 3, 2))
            g = gram(h)
            z = model.readout(g).squeeze(-1)
            chunk = {'logits': [z.numpy()], 'gram': [g.numpy()],
                     'post_logits': [z.numpy()], 'post_gram': [g.numpy()],
                     'selected_labels': [np.full(len(b), -1, np.int64)]}
            for t in range(1, length + 1):
                h = environment_step(model, h, u[:, t - 1], r[:, t], e[:, t - 1])
                g = gram(h)
                z = model.readout(g).squeeze(-1)
                chosen = np.full(len(b), -1, np.int64)
                if mask[t]:
                    if condition == 'GAUGE8':
                        x = h.numpy().astype(np.float64)
                        h = torch.from_numpy(root_state(x.transpose(0, 2, 1) @ x)[0])
                    else:
                        chosen = (z.numpy() >= 0).astype(np.int64)
                        h = proto[torch.from_numpy(chosen)]
                gp = gram(h) if mask[t] else g
                zp = model.readout(gp).squeeze(-1) if mask[t] else z
                for key, value in (('logits', z.numpy()), ('gram', g.numpy()),
                                   ('post_logits', zp.numpy()), ('post_gram', gp.numpy()),
                                   ('selected_labels', chosen)):
                    chunk[key].append(value)
            for name in names:
                batches[name].append(np.stack(chunk[name], 1))
    return {name: np.concatenate(batches[name]) for name in names} | {'refresh_mask': mask}


def metrics(z, target, command):
    z = z.astype(np.float64)
    oracle = np.bitwise_xor.accumulate(np.column_stack((target[:, 0].astype(np.int64), command)), axis=1)
    require(np.array_equal(target, oracle), 'XOR oracle')
    correct = (z >= 0) == target
    bad = ~correct
    ever = bad.any(1)
    first = np.argmax(bad, 1)
    loss = np.logaddexp(0, z) - z * target
    return {'bce': float(loss.mean()), 'future_bce': float(loss[:, 1:].mean()),
            'accuracy_all': float(correct.mean()), 'future_accuracy': float(correct[:, 1:].mean()),
            'final_accuracy': float(correct[:, -1].mean()),
            'initial_accuracy': float(correct[:, 0].mean()),
            'hold_accuracy': float(correct[:, 1:][command == 0].mean()) if np.any(command == 0) else None,
            'flip_accuracy': float(correct[:, 1:][command == 1].mean()) if np.any(command == 1) else None,
            'first_error_count': int(ever.sum()),
            'first_error_mean_time': float(first[ever].mean()) if ever.any() else None,
            'first_error_by_time': np.bincount(first[ever], minlength=z.shape[1]).tolist(),
            'accuracy_by_time': correct.mean(0).tolist()}


def same_metrics(actual, saved):
    require(actual.keys() == saved.keys(), 'metric keys')
    for key in actual:
        if isinstance(actual[key], list):
            close(actual[key], saved[key], key, 1e-12)
        elif actual[key] is None or isinstance(actual[key], int):
            require(actual[key] == saved[key], key)
        else:
            close(actual[key], saved[key], key, 1e-12)


def run(path):
    started = time.monotonic()
    manifest, completed, analysis = [load(path / name) for name in
                                     ('manifest.json', 'completion.json', 'analysis.json')]
    require(manifest['git_commit'] == '94f31a9d05593a8f75a6b25dad0de6650e6df98f', 'commit')
    require(load(ROOT / 'configs/exp0024_v1.json') == manifest['config'], 'config')
    require(digest(ROOT / 'configs/exp0024_v1.json') == manifest['config_sha256'], 'config hash')
    require({str(p.relative_to(ROOT)): digest(p) for p in sorted((ROOT / 'src').rglob('*.py'))
             if not p.name.startswith('._')} == manifest['source_sha256'], 'source hashes')
    for rel, sha in manifest['frozen_hashes'].items():
        require(digest(ROOT / rel) == sha, f'frozen hash {rel}')
    for rel, sha in manifest['parent_artifact_sha256'].items():
        require(digest(ROOT / rel) == sha, f'parent hash {rel}')
    require(digest(path / 'parent-manifest.json') == manifest['parent_artifact_sha256']['artifacts/EXP-0023/v2/manifest.json'], 'parent manifest copy')
    require(digest(path / 'parent-completion.json') == manifest['parent_artifact_sha256']['artifacts/EXP-0023/v2/completion.json'], 'parent completion copy')
    require(completed['status'] == 'COMPLETED' and completed['research_updates'] == 0 and
            completed['prototype_fits'] == 12 and completed['evaluation_count'] == 192 and
            completed['evaluation_transition_tokens'] == 12976128 and
            completed['calibration_transition_tokens'] == 49152, 'resource totals')
    require(digest(path / 'manifest.json') == analysis['manifest_sha256'] and
            digest(path / 'completion.json') == analysis['completion_sha256'], 'analysis provenance')
    require(set(completed['data_files']) == {f'data/{s}/{c}.npz' for s in SEEDS
             for c in ('calibration', *CELLS)}, 'input matrix')
    rows = {(r['seed'], r['arm']): r for r in completed['records']}
    require(set(rows) == {(s, a) for s in SEEDS for a in ARMS}, 'model matrix')
    checked = 0
    replay = {}
    for seed in SEEDS:
        datasets = {'calibration': generate(512, 8, [242001, seed, 8, 0], 'MIX', .4, 0.)}
        datasets.update({cell: generate(512, length, [243001, seed, length, code], mode, amplitude, 0.)
                         for cell, (length, mode, amplitude, code) in CELLS.items()})
        for cell, data in datasets.items():
            rel = f'data/{seed}/{cell}.npz'
            require(digest(path / rel) == completed['data_files'][rel], f'input hash {rel}')
            with np.load(path / rel) as stored:
                require(set(stored.files) == set(data), f'input keys {rel}')
                for key in data:
                    require(np.array_equal(stored[key], data[key]), f'input values {rel}:{key}')
        for arm in ARMS:
            row = rows[seed, arm]
            folder = path / str(seed) / arm
            require(load(folder / 'result.json') == row, 'row copy')
            for name, sha in row['artifact_sha256'].items():
                require(digest(folder / name) == sha, f'row hash {seed}/{arm}/{name}')
            require(row['calibration_data_sha256'] == completed['data_files'][f'data/{seed}/calibration.npz'], 'calibration link')
            model = model_at(folder / 'checkpoint-1000.pt', seed, arm)
            before = state_hash(model)
            require(before == row['frozen_model_state_sha256'], 'frozen state hash')
            z, _, h = predict_full(model, datasets['calibration'])
            labels = datasets['calibration']['targets'][:, -1].astype(np.int64)
            h64 = h.astype(np.float64)
            matrices = h64.transpose(0, 2, 1) @ h64
            means = np.stack([matrices[labels == b].mean(0) for b in (0, 1)])
            proto, eig = root_state(means)
            with np.load(folder / 'prototypes.npz') as stored:
                require(np.array_equal(stored['calibration_labels'], labels), 'calibration labels')
                close(stored['calibration_logits'], z, 'calibration forward', 1e-6)
                close(stored['calibration_states'], h, 'calibration states', 1e-6)
                close(stored['prototype_grams'], means, 'calibration means', 1e-12)
                close(stored['eigenvalues'], eig, 'PSD eigenvalues', 1e-12)
                close(stored['prototypes'], proto, 'PSD roots', 1e-7)
            pz = model.readout(gram(torch.from_numpy(proto))).squeeze(-1).detach().numpy()
            close(pz, np.load(folder / 'prototypes.npz')['prototype_logits'], 'prototype logits', 1e-6)
            cal = row['calibration']
            require(cal['calibration_counts'] == [int((labels == b).sum()) for b in (0, 1)], 'class counts')
            close(cal['calibration_accuracy'], np.mean((z[:, -1] >= 0) == labels), 'calibration accuracy', 1e-12)
            require(cal['prototype_labels_correct'] == bool(np.array_equal(pz >= 0, [False, True])), 'prototype labels')
            close(cal['min_eigenvalue'], eig.min(), 'minimum eigenvalue', 1e-12)
            require(cal['negative_eigenvalues_clamped'] == int((eig < 0).sum()), 'clamped eigenvalues')
            table = np.stack([model.readout(gram(model.transition(torch.from_numpy(proto),
                       torch.full((2,), u, dtype=torch.int64)))).squeeze(-1).detach().numpy() for u in (0, 1)], 1)
            with np.load(folder / 'transition_table.npz') as stored:
                close(table, stored['logits'], 'static logits', 1e-6)
                require(np.array_equal(table >= 0, stored['predictions']), 'static predictions')
                require(np.array_equal(stored['expected'], [[0, 1], [1, 0]]), 'static expected')
            resources = row['resources']
            require(resources['parameters'] == sum(p.numel() for p in model.parameters()) and
                    resources['parameter_bytes'] == sum(p.numel() * p.element_size() for p in model.parameters()) and
                    resources['prototype_values'] == 12 and resources['prototype_bytes'] == 48 and
                    resources['optimizer_updates'] == 0 and resources['prototype_fits'] == 1, 'model resources')
            replay[f'{seed}/{arm}'] = {'calibration_accuracy': cal['calibration_accuracy'],
                                      'static_transition_correct': bool(np.array_equal(table >= 0, [[0, 1], [1, 0]])),
                                      'cells': {}}
            for condition in CONDS:
                replay[f'{seed}/{arm}']['cells'][condition] = {}
                for cell, data in datasets.items():
                    if cell == 'calibration':
                        continue
                    record = row['evaluation'][condition][cell]
                    rawpath = folder / condition / f'{cell}.npz'
                    require(digest(rawpath) == record['artifact_sha256'], 'rollout hash')
                    with np.load(rawpath) as raw:
                        independently = independently_roll(model, data, condition, proto)
                        for key, value in independently.items():
                            if key in ('selected_labels', 'refresh_mask'):
                                require(np.array_equal(value, raw[key]), f'rollout {seed}/{arm}/{condition}/{cell}/{key}')
                            else:
                                close(value, raw[key], f'rollout {seed}/{arm}/{condition}/{cell}/{key}', 2e-6)
                        require(not raw['refresh_mask'][0], 't0 refresh')
                        if condition.startswith('RESET'):
                            mask = independently['refresh_mask']
                            require(np.array_equal(raw['selected_labels'][:, mask],
                                   (raw['logits'][:, mask] >= 0).astype(np.int64)), 'selected from pre logit')
                        elif condition == 'GAUGE8':
                            dz = np.max(np.abs(raw['post_logits'][:, raw['refresh_mask']] - raw['logits'][:, raw['refresh_mask']]))
                            dg = np.max(np.abs(raw['post_gram'][:, raw['refresh_mask']] - raw['gram'][:, raw['refresh_mask']]))
                            close(dz, record['diagnostics']['local_logit_abs_max'], 'gauge logit', 1e-7)
                            close(dg, record['diagnostics']['local_gram_abs_max'], 'gauge Gram', 1e-7)
                        result = metrics(independently['logits'], data['targets'], data['commands'])
                        same_metrics(result, record['metrics'])
                        require(record['transition_tokens'] == 512 * data['commands'].shape[1] and
                                record['refresh_calls'] == 512 * int(independently['refresh_mask'].sum()), 'cell resources')
                        replay[f'{seed}/{arm}']['cells'][condition][cell] = {
                            'final_accuracy': result['final_accuracy'], 'accuracy_all': result['accuracy_all'],
                            'flip_accuracy': result['flip_accuracy'], 'bce': result['bce'],
                            'first_error_count': result['first_error_count']}
                    checked += 1
            require(state_hash(model) == before, 'model weights changed')
    require(checked == 192, 'evaluation count')
    case = rows[23005, 'REL_BASE']
    def metric(cond, cell, key):
        return case['evaluation'][cond][cell]['metrics'][key]
    def long(cond):
        return np.mean([metric(cond, c, 'final_accuracy') for c in ('LONG_H', 'LONG_U')])
    base, gauge, reset, reset1 = (float(long(c)) for c in CONDS)
    qualifications = {
        'calibration_accuracy': case['calibration']['calibration_accuracy'] >= .95,
        'prototype_labels': case['calibration']['prototype_labels_correct'],
        'none_iid_h_all': metric('NONE', 'IID_H', 'accuracy_all') >= .95,
        'none_iid_u_all': metric('NONE', 'IID_U', 'accuracy_all') >= .95,
        'none_iid_u_flip': metric('NONE', 'IID_U', 'flip_accuracy') >= .95,
        'gauge_local': all(case['evaluation']['GAUGE8'][c]['diagnostics']['local_logit_abs_max'] <= 1e-5
                           and case['evaluation']['GAUGE8'][c]['diagnostics']['local_gram_abs_max'] <= 1e-5 for c in CELLS),
        'gauge_long_equivalence': abs(base - gauge) <= .005}
    if not all(qualifications.values()):
        decision = ('INCONCLUSIVE', 'CASE_QUALIFICATION_FAILURE')
    elif base < .95 and reset >= .95 and reset - gauge >= .10:
        decision = ('SUPPORTED', 'FROZEN_CASE_STATE_REFRESH_RESCUE')
    elif reset - gauge >= .05:
        decision = ('PARTIALLY_SUPPORTED', 'SMALLER_CASE_RESCUE')
    else:
        decision = ('INCONCLUSIVE', 'NO_REGISTERED_CASE_RESCUE')
    require(decision == (analysis['decision'], analysis['substatus']), 'decision')
    require(qualifications == analysis['case_qualification'], 'case qualification')
    for cond, value in zip(CONDS, (base, gauge, reset, reset1)):
        close(value, analysis['case_long_final_accuracy'][cond], 'case long accuracy', 1e-12)
    close(reset - gauge, analysis['case_reset8_minus_gauge8'], 'case effect', 1e-12)
    effects = {}
    for comparator in ('NONE', 'GAUGE8'):
        values = [float(np.mean([rows[s, 'REL_BASE']['evaluation']['RESET8'][c]['metrics']['final_accuracy'] -
                                 rows[s, 'REL_BASE']['evaluation'][comparator][c]['metrics']['final_accuracy']
                                 for c in ('LONG_H', 'LONG_U')])) for s in SEEDS]
        arr = np.asarray(values)
        mean, sd = float(arr.mean()), float(arr.std(ddof=1))
        half = 2.570581836 * sd / np.sqrt(6)
        effect = {'values_by_seed': {str(s): v for s, v in zip(SEEDS, values)}, 'mean': mean,
                  'sample_sd': sd, 'min': float(arr.min()), 'max': float(arr.max()),
                  'descriptive_t95': [mean - half, mean + half]}
        require(effect == analysis['exposed_rel_descriptive_effects'][f'RESET8_minus_{comparator}'], 'seed statistics')
        effects[comparator] = effect
    harms = []
    for seed in SEEDS:
        for arm in ARMS:
            item = replay[f'{seed}/{arm}']['cells']
            for cell in CELLS:
                for cond in ('RESET8', 'RESET1'):
                    delta = item[cond][cell]['final_accuracy'] - item['NONE'][cell]['final_accuracy']
                    if delta < 0:
                        harms.append({'seed': seed, 'arm': arm, 'condition': cond,
                                      'cell': cell, 'final_accuracy_delta': delta})
    elapsed = time.monotonic() - started
    require(elapsed <= 300, 'replay time cap')
    return {'status': 'PASS', 'decision': decision[0], 'substatus': decision[1],
            'case_qualification': qualifications,
            'case_long_final_accuracy': dict(zip(CONDS, (base, gauge, reset, reset1))),
            'case_reset8_minus_gauge8': reset - gauge, 'exposed_rel_effects': effects,
            'harms': harms, 'model_count': 12, 'prototype_fits': 12,
            'evaluation_count': checked, 'research_updates': 0,
            'evaluation_transition_tokens': 12976128, 'calibration_transition_tokens': 49152,
            'replay_seconds': elapsed, 'source_checkout': str(ROOT),
            'frozen_commit': manifest['git_commit'], 'original_run': str(path),
            'shared_code_boundary': 'Registered generator and frozen model transition/forward only; independent calibration algebra, intervention rollout, metrics and decision.'}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    started = time.monotonic()
    try:
        result = run(args.run)
    except BaseException as exc:
        result = {'status': 'FAIL', 'error': repr(exc), 'elapsed_seconds': time.monotonic() - started}
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
        raise
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: result[k] for k in ('status', 'decision', 'substatus', 'replay_seconds', 'evaluation_count')}))


if __name__ == '__main__':
    main()
