"""Independent, single-pass EXP-0025/v2 result replay; no training or raw writes.

Run only after the formal run has completed, from frozen commit 1305b15 with
PYTHONPATH=<frozen>/src and /private/tmp/neurotopo-venv/bin/python. The shared
pieces are the registered model class, generator, and tensor/Gram primitives.
Interventions, calibration, metrics, geometry audits and R/M decisions below
are independently implemented. The replay has a hard 600 second budget.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import signal
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from neurotopo.learned_protected_pilot import LearnedMemory, generate, gram, radial, state_hash

SEEDS = tuple(range(25001, 25013))
ALL_SEEDS = (*SEEDS, 23005)
ARMS = ('REL', 'INV')
METHODS = ('NONE', 'GAUGE8', 'FULL8', 'EQUAL8', 'SCALE8')
CELLS = {'IID_H': (8, 'HOLD', .4, 1), 'IID_U': (8, 'UPDATE', .4, 2),
         'LONG_H': (256, 'HOLD', np.pi, 1), 'LONG_U': (256, 'UPDATE', np.pi, 2)}
STEPS = (0, 125, 250, 500, 1000)
COMMIT = '1305b1516eaa38dd97f037b4d2e10a14d3ba2553'
START = time.monotonic()
CAP = 600


def budget() -> None:
    if time.monotonic() - START >= CAP:
        raise TimeoutError('Single independent replay exceeded 600 seconds')


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as f:
        return {k: f[k].copy() for k in f.files}


def same(actual, expected, name: str, discrepancies: list, tol: float = 0.) -> None:
    a, b = np.asarray(actual), np.asarray(expected)
    if a.shape != b.shape or a.dtype.kind != b.dtype.kind:
        discrepancies.append({'name': name, 'reason': 'shape_or_type', 'actual': str((a.shape, a.dtype)),
                              'expected': str((b.shape, b.dtype))})
        return
    if a.dtype.kind in 'f':
        diff = float(np.max(np.abs(a.astype(float) - b.astype(float)))) if a.size else 0.
        if not np.isfinite(diff) or diff > tol:
            discrepancies.append({'name': name, 'maximum_abs_error': diff, 'tolerance': tol})
    elif not np.array_equal(a, b):
        discrepancies.append({'name': name, 'reason': 'value_mismatch'})


def compare_structure(a, b, path: str, discrepancies: list, tol=1e-12) -> None:
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a) != set(b):
            discrepancies.append({'name': path, 'reason': 'keys', 'actual': sorted(a), 'expected': sorted(b)})
        for k in set(a) & set(b):
            compare_structure(a[k], b[k], path + '/' + k, discrepancies, tol)
    elif a is None or b is None or isinstance(a, (str, bool)) or isinstance(b, (str, bool)):
        if a != b:
            discrepancies.append({'name': path, 'actual': a, 'expected': b})
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if not math.isclose(a, b, rel_tol=0., abs_tol=tol):
            discrepancies.append({'name': path, 'actual': a, 'expected': b})
    elif isinstance(a, list) and isinstance(b, list):
        same(a, b, path, discrepancies, tol)
    elif a != b:
        discrepancies.append({'name': path, 'actual': a, 'expected': b})


def model_at(path: Path, seed: int, arm: str, step=1000) -> LearnedMemory:
    cp = torch.load(path, map_location='cpu', weights_only=True)
    expected_variant = arm + '_BASE' if seed == 23005 else arm
    if (cp['seed'], cp['arm'], cp['variant'], cp['step']) != (seed, arm, expected_variant, step):
        raise RuntimeError(f'Checkpoint metadata mismatch: {path}')
    model = LearnedMemory(arm, seed=seed)
    model.load_state_dict(cp['model'])
    model.eval()
    return model


def sigmoid_radial_initial(model: LearnedMemory, data: dict, start: int, stop: int) -> torch.Tensor:
    b = torch.from_numpy(data['b0'][start:stop])
    r = torch.from_numpy(data['common'][start:stop, 0])
    return r @ radial(model.encoder(b).reshape(-1, 3, 2))


def step_state(model: LearnedMemory, h: torch.Tensor, data: dict, start: int, stop: int, t: int) -> torch.Tensor:
    r = torch.from_numpy(data['common'][start:stop, t + 1])
    e = torch.from_numpy(data['leak'][start:stop, t])
    u = torch.from_numpy(data['commands'][start:stop, t])
    distorted = r @ h
    second = (e @ distorted[:, :, 1:2]).squeeze(-1)
    return model.transition(torch.stack((distorted[:, :, 0], second), dim=-1), u)


def root_state(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    sym = (matrix.astype(np.float64) + matrix.astype(np.float64).swapaxes(-1, -2)) / 2
    values, vectors = np.linalg.eigh(sym)
    if values.min() < -1e-8 or not np.isfinite(values).all():
        raise ValueError('Non-PSD Gram in independent root')
    root = (vectors * np.sqrt(np.maximum(values, 0))[..., None, :]) @ vectors.swapaxes(-1, -2)
    out = np.zeros((*root.shape[:-2], 3, 2), np.float32)
    out[..., :2, :] = root
    return out, values


def forward_final(model: LearnedMemory, data: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    zs, gs, hs = [], [], []
    with torch.no_grad():
        for start in range(0, len(data['b0']), 128):
            stop = min(start + 128, len(data['b0']))
            h = sigmoid_radial_initial(model, data, start, stop)
            g = gram(h)
            z = model.readout(g).squeeze(-1)
            zparts, gparts = [z.numpy()], [g.numpy()]
            for t in range(data['commands'].shape[1]):
                h = step_state(model, h, data, start, stop, t)
                g = gram(h)
                z = model.readout(g).squeeze(-1)
                zparts.append(z.numpy()); gparts.append(g.numpy())
            zs.append(np.stack(zparts, 1)); gs.append(np.stack(gparts, 1)); hs.append(h.numpy())
    return np.concatenate(zs), np.concatenate(gs), np.concatenate(hs)


def calibrate(model: LearnedMemory, data: dict) -> dict:
    z, g, h = forward_final(model, data)
    labels = data['targets'][:, -1].astype(np.int64)
    if set(labels) != {0, 1}:
        raise RuntimeError('Calibration missing a label')
    matrix = h.astype(float).transpose(0, 2, 1) @ h.astype(float)
    means = np.stack([matrix[labels == k].mean(0) for k in (0, 1)])
    prototypes, eigenvalues = root_state(means)
    with torch.no_grad():
        pz = model.readout(gram(torch.from_numpy(prototypes))).squeeze(-1).numpy()
    return {'prototypes': prototypes, 'prototype_grams': means, 'prototype_logits': pz,
            'calibration_logits': z, 'calibration_gram': g, 'calibration_states': h,
            'calibration_labels': labels, 'eigenvalues': eigenvalues}


def weighted(g: np.ndarray) -> np.ndarray:
    out = np.asarray(g, float).copy()
    out[..., 1] *= math.sqrt(2.)
    return out


def normalized(g: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    v = weighted(g)
    norm = np.linalg.norm(v, axis=-1)
    return v / np.maximum(norm[..., None], 1e-12), norm


def intervene(model: LearnedMemory, h: torch.Tensor, z: torch.Tensor,
              condition: str, proto: torch.Tensor) -> tuple[torch.Tensor, dict]:
    pre_g = gram(h).numpy().astype(float)
    pre_z = z.numpy().astype(float)
    labels = (pre_z >= 0).astype(np.int64)
    picked = proto[torch.from_numpy(labels)]
    p_g = gram(picked).numpy().astype(float)
    p_z = model.readout(gram(picked)).squeeze(-1).numpy().astype(float)
    q = pre_z - float(model.readout.bias[0])
    qp = p_z - float(model.readout.bias[0])
    reason = np.zeros(len(h), np.int64)
    ratio = np.ones(len(h), float)
    available = np.zeros(len(h), bool)
    selected = labels if condition != 'GAUGE8' else np.full(len(h), -1, np.int64)
    if condition == 'GAUGE8':
        candidate = torch.from_numpy(root_state(h.numpy().astype(float).transpose(0, 2, 1) @ h.numpy().astype(float))[0])
    elif condition == 'FULL8':
        candidate = picked
    else:
        denom, numer = (qp, q) if condition == 'EQUAL8' else (q, qp)
        available = np.abs(denom) > 1e-6
        reason[~available] = 1
        with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
            ratio = np.divide(numer, denom, out=np.ones(len(h)), where=available)
        reason[available & (~np.isfinite(ratio) | (ratio <= 0))] = 2
        available &= np.isfinite(ratio)
        base = picked if condition == 'EQUAL8' else h
        with np.errstate(invalid='ignore', over='ignore'):
            proposed = (base.numpy().astype(float) * np.sqrt(np.where(reason == 0, ratio, 1.))[:, None, None]).astype(np.float32)
        reason[(reason == 0) & ~np.isfinite(proposed).all(axis=(1, 2))] = 3
        proposed = np.where((reason == 0)[:, None, None], proposed, h.numpy())
        candidate = torch.from_numpy(proposed)
        c_g = gram(candidate)
        reason[(reason == 0) & ~torch.isfinite(c_g).all(1).numpy()] = 3
        candidate = torch.where(torch.from_numpy(reason == 0)[:, None, None], candidate, h)
        c_g = gram(candidate).numpy()
        c_z = model.readout(gram(candidate)).squeeze(-1).numpy().astype(float)
        reason[(reason == 0) & ~np.isfinite(c_z)] = 3
        target_g = p_g if condition == 'EQUAL8' else pre_g
        sh, n1 = normalized(c_g); ref, n2 = normalized(target_g)
        reason[(reason == 0) & ((n1 < 1e-12) | (n2 < 1e-12))] = 4
        target_z = pre_z if condition == 'EQUAL8' else p_z
        reason[(reason == 0) & (np.abs(c_z - target_z) > 1e-5)] = 5
        reason[(reason == 0) & (np.linalg.norm(sh - ref, axis=1) > 1e-5)] = 6
        if condition == 'EQUAL8':
            reason[(reason == 0) & ((c_z >= 0) != (pre_z >= 0))] = 7
        candidate = torch.where(torch.from_numpy(reason == 0)[:, None, None], candidate, h)
    post_g = gram(candidate).numpy().astype(float)
    post_z = model.readout(gram(candidate)).squeeze(-1).numpy().astype(float)
    target_g = p_g if condition in ('FULL8', 'EQUAL8') else pre_g
    target_z = pre_z if condition in ('GAUGE8', 'EQUAL8') else p_z
    sh, _ = normalized(post_g); ref, _ = normalized(target_g)
    if condition == 'GAUGE8':
        reason[(np.abs(post_z - pre_z) > 1e-5) | (np.max(np.abs(post_g - pre_g), axis=1) > 1e-5)] = 5
    diag = {'selected_labels': selected, 'valid_event': reason == 0, 'invalid_reason': reason,
            'q_current': q, 'q_proto': qp, 'proposed_scale': np.where(np.isfinite(ratio), ratio, 0.),
            'ratio_available': available, 'logit_target_error': np.abs(post_z - target_z),
            'shape_error': np.linalg.norm(sh - ref, axis=1),
            'state_delta_norm': torch.linalg.vector_norm((candidate - h).flatten(1), dim=1).numpy(),
            'gram_delta_norm': np.linalg.norm(weighted(post_g - pre_g), axis=1),
            'prototype_distance_before': np.linalg.norm(weighted(pre_g - p_g), axis=1),
            'prototype_distance_after': np.linalg.norm(weighted(post_g - p_g), axis=1),
            'post_state_norm': torch.linalg.vector_norm(candidate.flatten(1), dim=1).numpy()}
    return candidate, diag


def replay(model: LearnedMemory, data: dict, condition: str, prototypes: np.ndarray) -> dict:
    n, length = data['commands'].shape
    mask = np.zeros(length + 1, bool)
    if condition != 'NONE':
        mask[8::8] = True
    proto = torch.from_numpy(prototypes.astype(np.float32))
    keys = ('selected_labels', 'valid_event', 'invalid_reason', 'q_current', 'q_proto',
            'proposed_scale', 'ratio_available', 'logit_target_error', 'shape_error',
            'state_delta_norm', 'gram_delta_norm', 'prototype_distance_before',
            'prototype_distance_after', 'post_state_norm')
    collected = {}
    with torch.no_grad():
        for start in range(0, n, 128):
            stop = min(start + 128, n)
            h = sigmoid_radial_initial(model, data, start, stop)
            z = model.readout(gram(h)).squeeze(-1)
            local = {'logits': [z.numpy()], 'gram': [gram(h).numpy()],
                     'post_logits': [z.numpy()], 'post_gram': [gram(h).numpy()]}
            def blank() -> dict:
                return {k: np.full(len(h), -1, np.int64) if k == 'selected_labels'
                        else np.zeros(len(h), bool) if k in ('valid_event', 'ratio_available')
                        else np.zeros(len(h), np.int64) if k == 'invalid_reason'
                        else np.zeros(len(h), float) for k in keys}
            for k, v in blank().items():
                local[k] = [v]
            for t in range(length):
                h = step_state(model, h, data, start, stop, t)
                g = gram(h)
                z = model.readout(g).squeeze(-1)
                local['logits'].append(z.numpy()); local['gram'].append(g.numpy())
                if mask[t + 1]:
                    h, details = intervene(model, h, z, condition, proto)
                    pg = gram(h); pz = model.readout(pg).squeeze(-1)
                else:
                    details = blank(); pg, pz = g, z
                local['post_logits'].append(pz.numpy()); local['post_gram'].append(pg.numpy())
                for k, v in details.items():
                    local[k].append(v)
            for k, v in local.items():
                collected.setdefault(k, []).append(np.stack(v, axis=1))
    result = {k: np.concatenate(v, axis=0) for k, v in collected.items()}
    result['refresh_mask'] = mask
    return result


def metrics(raw: dict, data: dict) -> dict:
    z = raw['logits'].astype(float)
    target = data['targets']
    command = data['commands']
    oracle = np.bitwise_xor.accumulate(np.column_stack((target[:, 0].astype(int), command)), axis=1)
    if not np.array_equal(target, oracle):
        raise RuntimeError('Target differs from XOR oracle')
    correct = (z >= 0) == target
    wrong = ~correct
    first = np.argmax(wrong, axis=1)
    ever = wrong.any(axis=1)
    hold, flip = command == 0, command == 1
    loss = np.logaddexp(0., z) - z * target
    return {'bce': float(loss.mean()), 'future_bce': float(loss[:, 1:].mean()),
            'accuracy_all': float(correct.mean()), 'future_accuracy': float(correct[:, 1:].mean()),
            'final_accuracy': float(correct[:, -1].mean()), 'initial_accuracy': float(correct[:, 0].mean()),
            'hold_accuracy': float(correct[:, 1:][hold].mean()) if hold.any() else None,
            'flip_accuracy': float(correct[:, 1:][flip].mean()) if flip.any() else None,
            'first_error_count': int(ever.sum()), 'first_error_mean_time': float(first[ever].mean()) if ever.any() else None,
            'first_error_by_time': np.bincount(first[ever], minlength=z.shape[1]).tolist(),
            'accuracy_by_time': correct.mean(axis=0).tolist()}


def diagnostics(raw: dict, method: str) -> dict:
    mask = raw['refresh_mask'].astype(bool)
    z, pz = raw['logits'], raw['post_logits']
    g, pg = raw['gram'], raw['post_gram']
    selected = raw['selected_labels']
    expected = np.zeros(z.shape[1], bool)
    if method != 'NONE':
        expected[8::8] = True
    if not np.array_equal(mask, expected) or np.any(selected[:, ~mask] != -1):
        raise RuntimeError('Wrong refresh schedule or unscheduled choice')
    if not np.array_equal(z[:, ~mask], pz[:, ~mask]) or not np.array_equal(g[:, ~mask], pg[:, ~mask]):
        raise RuntimeError('Unscheduled mutation')
    if method in ('NONE', 'GAUGE8'):
        if np.any(selected != -1):
            raise RuntimeError('Control chose prototype')
    elif not np.array_equal(selected[:, mask], (z[:, mask] >= 0).astype(int)):
        raise RuntimeError('Prototype choice uses more than current sign')
    valid = raw['valid_event'][:, mask]
    reason = raw['invalid_reason'][:, mask]
    if not np.array_equal(valid, reason == 0):
        raise RuntimeError('Event reason mismatch')
    dz = np.abs(pz[:, mask].astype(float) - z[:, mask].astype(float))
    dg = np.abs(pg[:, mask].astype(float) - g[:, mask].astype(float))
    gram_delta = np.linalg.norm(weighted(pg[:, mask].astype(float) - g[:, mask].astype(float)), axis=-1)
    if method in ('EQUAL8', 'SCALE8'):
        if not np.array_equal(z[:, mask][~valid], pz[:, mask][~valid]) or not np.array_equal(g[:, mask][~valid], pg[:, mask][~valid]):
            raise RuntimeError('Invalid proposal did not fall back')
    def maximum(x):
        return float(np.max(x)) if np.size(x) else None
    def mean(x):
        return float(np.mean(x)) if np.size(x) else None
    count = int(valid.size)
    available = raw['ratio_available'][:, mask]
    values = raw['proposed_scale'][:, mask][available]
    return {'refresh_calls': count, 'valid_count': int(valid.sum()),
            'valid_fraction': float(valid.mean()) if count else None,
            'invalid_reasons': {str(i): int((reason == i).sum()) for i in range(1, 8)},
            'local_logit_abs_max': maximum(dz), 'local_gram_abs_max': maximum(dg),
            'valid_target_logit_error_max': maximum(raw['logit_target_error'][:, mask][valid]),
            'valid_shape_error_max': maximum(raw['shape_error'][:, mask][valid]),
            'valid_sign_changes': int(((z[:, mask] >= 0) != (pz[:, mask] >= 0))[valid].sum()),
            'valid_nonzero_gram_fraction': mean(gram_delta[valid] > 1e-6),
            'gram_change_mean': mean(gram_delta), 'gram_change_max': maximum(gram_delta),
            'state_change_mean': mean(raw['state_delta_norm'][:, mask]),
            'state_change_max': maximum(raw['state_delta_norm'][:, mask]),
            'prototype_distance_before_mean': mean(raw['prototype_distance_before'][:, mask]),
            'prototype_distance_after_mean': mean(raw['prototype_distance_after'][:, mask]),
            'available_ratio_count': int(available.sum()),
            'ratio_range': [float(values.min()), float(values.max())] if len(values) else None}


def rate(k: int, n: int) -> dict:
    if not n:
        return {'count': k, 'denominator': n, 'rate': None, 'wilson95': None}
    z = 1.959963985
    p = k / n
    divisor = 1 + z*z/n
    centre = (p + z*z/(2*n)) / divisor
    half = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))/divisor
    return {'count': k, 'denominator': n, 'rate': p,
            'wilson95': [max(0., centre-half), min(1., centre+half)]}


def long_accuracy(row: dict, condition: str, phase='evaluation') -> float:
    return float(np.mean([row[phase][condition][cell]['metrics']['final_accuracy']
                          for cell in ('LONG_H', 'LONG_U')]))


def independent_decision(rows: list[dict]) -> dict:
    lookup = {(r['seed'], r['arm']): r for r in rows}
    if len(lookup) != 26 or set(lookup) != {(s, arm) for s in ALL_SEEDS for arm in ARMS}:
        raise RuntimeError('Incomplete 26-model matrix')
    seed_rows = []
    for seed in SEEDS:
        row = lookup[seed, 'REL']
        cal = row['calibration']; qual = row['qualification']['NONE']
        qualified = bool(cal['calibration_accuracy'] >= .95 and cal['prototype_labels_correct']
                         and qual['IID_H']['metrics']['accuracy_all'] >= .95
                         and qual['IID_U']['metrics']['accuracy_all'] >= .95
                         and qual['IID_U']['metrics']['flip_accuracy'] >= .95)
        aq = long_accuracy(row, 'NONE', 'qualification')
        a = {c: long_accuracy(row, c) for c in METHODS}
        gauge = all(row['evaluation']['GAUGE8'][cell]['diagnostics'][k] <= 1e-5
                    for cell in CELLS for k in ('local_logit_abs_max', 'local_gram_abs_max'))
        gauge &= abs(a['GAUGE8'] - a['NONE']) <= .005
        stratum = 'FAILURE' if qualified and aq < .95 else 'SUCCESS' if qualified else 'UNQUALIFIED'
        d = [row['evaluation']['EQUAL8'][cell]['diagnostics'] for cell in ('LONG_H', 'LONG_U')]
        feasible = all(x['valid_fraction'] >= .99 and x['valid_target_logit_error_max'] is not None
                       and x['valid_target_logit_error_max'] <= 1e-5 and x['valid_sign_changes'] == 0 for x in d)
        changed = all(x['valid_nonzero_gram_fraction'] is not None
                      and x['valid_nonzero_gram_fraction'] >= .5 for x in d)
        exclusion = ('NOT_QUALIFICATION_FAILURE' if stratum != 'FAILURE' else 'GAUGE' if not gauge
                     else 'EQUAL_FEASIBILITY' if not feasible else 'INSUFFICIENT_CHANGE' if not changed else None)
        seed_rows.append({'seed': seed, 'qualified': qualified, 'stratum': stratum,
                          'qualification_A': aq, 'final_A': a, 'gauge_pass': gauge,
                          'M_exclusion': exclusion, 'baseline_failure_reproduced': a['NONE'] < .95,
                          'repair': {c: bool(a['NONE'] < .95 and a[c] >= .95 and a[c]-a['NONE'] >= .10)
                                     for c in METHODS[2:]},
                          'harm': {c: bool(a[c] < .95 and a[c]-a['NONE'] < -.02)
                                   for c in METHODS[2:]},
                          'delta': {c: a[c]-a['NONE'] for c in METHODS[1:]}})
    failures = [x for x in seed_rows if x['stratum'] == 'FAILURE']
    successes = [x for x in seed_rows if x['stratum'] == 'SUCCESS']
    eligible = [x for x in failures if x['M_exclusion'] is None]
    methods = {c: {'repair_original_failures': rate(sum(x['repair'][c] for x in failures), len(failures)),
                   'harm_qualified_successes': rate(sum(x['harm'][c] for x in successes), len(successes)),
                   'all_negative_deltas': {str(x['seed']): x['delta'][c] for x in seed_rows if x['delta'][c] < 0}}
               for c in METHODS[2:]}
    all_gauge = all(x['gauge_pass'] for x in seed_rows if x['qualified'])
    full_rate = methods['FULL8']['repair_original_failures']['rate']
    R = ('R_INCONCLUSIVE_FAILURE_COUNT' if len(failures) < 2 else
         'R_SUPPORTED_IN_NEW_BATCH' if full_rate >= .8 and methods['FULL8']['harm_qualified_successes']['count'] == 0 and all_gauge
         else 'R_NOT_SUPPORTED')
    m_rate = rate(sum(x['repair']['EQUAL8'] for x in eligible), len(eligible))
    M = ('M_INCONCLUSIVE_ELIGIBILITY' if len(eligible) < 2 else
         'M_SUPPORTED_WITHIN_EQUAL_LOGIT_FAMILY' if m_rate['rate'] >= .8 else 'M_NOT_SUPPORTED')
    rs, ms = R == 'R_SUPPORTED_IN_NEW_BATCH', M == 'M_SUPPORTED_WITHIN_EQUAL_LOGIT_FAMILY'
    effects = {}
    for arm in ARMS:
        effects[arm] = {}
        for condition in METHODS[1:]:
            values = np.array([long_accuracy(lookup[s, arm], condition)-long_accuracy(lookup[s, arm], 'NONE') for s in SEEDS])
            avg = float(values.mean()); sd = float(values.std(ddof=1)); half = 2.200985160*sd/math.sqrt(12)
            effects[arm][condition] = {'values': values.tolist(), 'mean': avg, 'sample_sd': sd,
                                       'min': float(values.min()), 'max': float(values.max()),
                                       'descriptive_t95': [avg-half, avg+half]}
    return {'decision': 'SUPPORTED' if rs and ms else 'PARTIALLY_SUPPORTED' if rs or ms else 'INCONCLUSIVE',
            'substatus': 'NEW_SEED_REFRESH_AND_EQUAL_LOGIT_RESCUE' if rs and ms else
            'R_ONLY_SUPPORTED' if rs else 'M_ONLY_SUPPORTED' if ms else 'NEITHER_ENDPOINT_SUPPORTED',
            'R': R, 'M': M, 'qualified_failure_seeds': [x['seed'] for x in failures],
            'qualified_success_seeds': [x['seed'] for x in successes],
            'M_eligible_seeds': [x['seed'] for x in eligible], 'M_conditional_repair': m_rate,
            'all_qualified_gauge_pass': all_gauge, 'seed_decisions': seed_rows,
            'methods': methods, 'new_seed_effects': effects}


def data_for(root: Path, seed: int, phase: str, cell: str | None, discrepancies: list) -> dict:
    if phase == 'monitor':
        generated = generate(128, 8, [251001, seed, 8, 0], 'MIX', .4, 0.)
        path = root / 'data' / str(seed) / 'monitor.npz'
    elif phase == 'calibration':
        generated = generate(512, 8, [253001, seed, 8, 0], 'MIX', .4, 0.)
        path = root / 'data' / str(seed) / 'calibration.npz'
    else:
        length, mode, amplitude, mode_id = CELLS[cell]
        namespace = 254001 if phase == 'qual' else 255001
        generated = generate(512, length, [namespace, seed, length, mode_id], mode, amplitude, 0.)
        path = root / 'data' / str(seed) / f'{phase}_{cell}.npz'
    saved = load_npz(path)
    if set(saved) != set(generated):
        discrepancies.append({'name': str(path), 'reason': 'input_columns'})
    for k in set(saved) & set(generated):
        same(saved[k], generated[k], f'{path}:{k}', discrepancies)
    return saved


def execute(root: Path, frozen: Path, report_dir: Path) -> tuple[dict, dict, dict]:
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    discrepancies = []
    inventory = {'monitor_checkpoints': 0, 'prototype_fits': 0, 'qualification_cells': 0,
                 'test_cells': 0, 'input_files': 0, 'raw_files': 0, 'model_states': 0}
    manifest = read(root / 'manifest.json')
    completion = read(root / 'completion.json')
    if manifest['git_commit'] != COMMIT or manifest['git_status'] != '':
        raise RuntimeError('Wrong formal source commit or dirty formal source')
    if completion['status'] != 'COMPLETED':
        raise RuntimeError('Formal run has not completed')
    if manifest['actual_output'] != str(root):
        raise RuntimeError('Run output path differs from manifest')
    if torch.__version__.split('+')[0] != '2.8.0' or np.__version__ != '2.3.3':
        raise RuntimeError('Wrong frozen library versions')
    for relative, expected in manifest['source_sha256'].items():
        if digest(frozen / relative) != expected:
            discrepancies.append({'name': relative, 'reason': 'source_hash'})
    for field in ('frozen_hashes', 'parent_artifact_sha256'):
        for relative, expected in manifest[field].items():
            if digest(frozen / relative) != expected:
                discrepancies.append({'name': relative, 'reason': field})
    suite = read(frozen / 'artifacts/validation/suite.json')
    preflight = read(frozen / 'artifacts/validation/exp0025_preflight.json')
    if suite.get('pass') is not True or preflight.get('pass') is not True:
        discrepancies.append({'name': 'validation', 'reason': 'not_passing'})
    if suite.get('source_sha256') != manifest['source_sha256'] or preflight.get('source_sha256') != manifest['source_sha256']:
        discrepancies.append({'name': 'validation', 'reason': 'source_mismatch'})
    if preflight.get('config_sha256') != manifest['config_sha256']:
        discrepancies.append({'name': 'validation', 'reason': 'config_mismatch'})
    for relative, expected in completion['data_files'].items():
        if digest(root / relative) != expected:
            discrepancies.append({'name': relative, 'reason': 'data_hash'})
    if len(completion['records']) != 26:
        raise RuntimeError('Wrong completion model count')
    lookup = {(r['seed'], r['arm']): r for r in completion['records']}
    if set(lookup) != {(s, a) for s in ALL_SEEDS for a in ARMS}:
        raise RuntimeError('Wrong model matrix')
    replay_rows = []
    for seed in ALL_SEEDS:
        budget()
        monitor = data_for(root, seed, 'monitor', None, discrepancies) if seed != 23005 else None
        if monitor is not None: inventory['input_files'] += 1
        calibration = data_for(root, seed, 'calibration', None, discrepancies)
        inventory['input_files'] += 1
        datasets = {}
        for cell in CELLS:
            datasets[('qual', cell)] = data_for(root, seed, 'qual', cell, discrepancies)
            datasets[('test', cell)] = data_for(root, seed, 'test', cell, discrepancies)
            inventory['input_files'] += 2
        for arm in ARMS:
            budget()
            source = root / str(seed) / arm
            row = lookup[seed, arm]
            if read(source / 'result.json') != row:
                discrepancies.append({'name': f'{seed}/{arm}', 'reason': 'result_completion_mismatch'})
            for relative, expected in row['artifact_sha256'].items():
                if digest(source / relative) != expected:
                    discrepancies.append({'name': f'{seed}/{arm}/{relative}', 'reason': 'artifact_hash'})
            model = model_at(source / 'checkpoint-1000.pt', seed, arm)
            inventory['model_states'] += 1
            if state_hash(model) != row['frozen_model_state_sha256']:
                discrepancies.append({'name': f'{seed}/{arm}', 'reason': 'model_state_hash'})
            if seed != 23005:
                for step in STEPS:
                    budget()
                    checkpoint = source / f'checkpoint-{step}.pt'
                    curve = next(c for c in row['curves'] if c['step'] == step)
                    if digest(checkpoint) != curve['checkpoint_sha256']:
                        discrepancies.append({'name': str(checkpoint), 'reason': 'checkpoint_hash'})
                    m = model_at(checkpoint, seed, arm, step)
                    z, g, _ = forward_final(m, monitor)
                    saved = load_npz(source / f'monitor-{step}.npz')
                    if digest(source / f'monitor-{step}.npz') != curve['prediction_sha256']:
                        discrepancies.append({'name': f'{seed}/{arm}/monitor-{step}', 'reason': 'prediction_hash'})
                    same(z, saved['logits'], f'{seed}/{arm}/monitor-{step}/logits', discrepancies)
                    same(g, saved['gram'], f'{seed}/{arm}/monitor-{step}/gram', discrepancies)
                    compare_structure(metrics({'logits': z}, monitor), curve['metrics'],
                                      f'{seed}/{arm}/monitor-{step}/metrics', discrepancies)
                    inventory['monitor_checkpoints'] += 1
            independent = calibrate(model, calibration)
            saved_p = load_npz(source / 'prototypes.npz')
            for k in independent:
                same(independent[k], saved_p[k], f'{seed}/{arm}/calibration/{k}', discrepancies, 1e-12)
            labels = independent['calibration_labels']
            summary = {'calibration_accuracy': float(np.mean((independent['calibration_logits'][:, -1] >= 0) == labels)),
                       'prototype_labels_correct': bool(np.array_equal(independent['prototype_logits'] >= 0, [False, True])),
                       'min_eigenvalue': float(independent['eigenvalues'].min()),
                       'negative_eigenvalues_clamped': int(np.sum(independent['eigenvalues'] < 0)),
                       'calibration_counts': [int(np.sum(labels == k)) for k in (0, 1)],
                       'prototype_values': 12, 'prototype_bytes': 48}
            compare_structure(summary, row['calibration'], f'{seed}/{arm}/calibration_summary', discrepancies)
            inventory['prototype_fits'] += 1
            replay_row = {'seed': seed, 'arm': arm, 'calibration': summary,
                          'qualification': {'NONE': {}}, 'evaluation': {c: {} for c in METHODS}}
            for phase in ('qual', 'test'):
                for condition in (('NONE',) if phase == 'qual' else METHODS):
                    for cell in CELLS:
                        budget()
                        data = datasets[phase, cell]
                        raw = replay(model, data, condition, independent['prototypes'])
                        saved_path = source / (f'qual_{cell}.npz' if phase == 'qual' else f'test/{condition}/{cell}.npz')
                        saved = load_npz(saved_path)
                        if set(raw) != set(saved):
                            discrepancies.append({'name': str(saved_path), 'reason': 'raw_columns'})
                        for k in set(raw) & set(saved):
                            same(raw[k], saved[k], f'{seed}/{arm}/{phase}/{condition}/{cell}/{k}', discrepancies, 1e-12)
                        independently_scored = {'metrics': metrics(raw, data), 'diagnostics': diagnostics(raw, condition)}
                        recorded = row['qualification']['NONE'][cell] if phase == 'qual' else row['evaluation'][condition][cell]
                        if digest(saved_path) != recorded['artifact_sha256']:
                            discrepancies.append({'name': str(saved_path), 'reason': 'raw_artifact_hash'})
                        compare_structure(independently_scored, {k: recorded[k] for k in independently_scored},
                                          f'{seed}/{arm}/{phase}/{condition}/{cell}/summary', discrepancies)
                        replay_row['qualification']['NONE'][cell] = independently_scored if phase == 'qual' else replay_row['qualification']['NONE'].get(cell)
                        if phase == 'test':
                            replay_row['evaluation'][condition][cell] = independently_scored
                        inventory['raw_files'] += 1
                        inventory['qualification_cells' if phase == 'qual' else 'test_cells'] += 1
            replay_rows.append(replay_row)
    decision = independent_decision(replay_rows)
    formal = read(root / 'analysis.json')
    for k in decision:
        if k not in formal:
            discrepancies.append({'name': 'analysis/' + k, 'reason': 'missing'})
        else:
            compare_structure(decision[k], formal[k], 'analysis/' + k, discrepancies)
    expected = {'research_updates': 24000, 'trained_models': 24, 'reference_models': 2,
                'model_count': 26, 'prototype_fits': 26, 'qualification_count': 104,
                'evaluation_count': 520, 'monitor_checkpoints': 120,
                'training_transition_tokens': 8256000, 'calibration_transition_tokens': 106496,
                'evaluation_transition_tokens': 42172416}
    for k, v in expected.items():
        if completion.get(k) != v:
            discrepancies.append({'name': 'completion/' + k, 'actual': completion.get(k), 'expected': v})
    for k, v in {'monitor_checkpoints': 120, 'prototype_fits': 26,
                 'qualification_cells': 104, 'test_cells': 520, 'raw_files': 624}.items():
        if inventory[k] != v:
            discrepancies.append({'name': 'replay/' + k, 'actual': inventory[k], 'expected': v})
    reproduction = {'status': 'PASS' if not discrepancies else 'MISMATCH',
                    'frozen_commit': COMMIT, 'script_sha256': digest(Path(__file__)),
                    'formal_output': str(root), 'wall_seconds': time.monotonic() - START,
                    'cap_seconds': CAP, 'inventory': inventory,
                    'discrepancies': discrepancies, 'shared_components':
                    ['LearnedMemory', 'generate', 'gram', 'radial', 'state_hash'],
                    'independent_components': ['transition orchestration', 'intervention', 'calibration',
                                               'metrics', 'geometry audit', 'R/M decisions'],
                    'no_optimizer_updates': True}
    audit = {'status': 'PASS' if not discrepancies else 'MISMATCH',
             'formal_manifest_commit': manifest['git_commit'], 'source_hash_count': len(manifest['source_sha256']),
             'formal_config_sha256': manifest['config_sha256'],
             'source_hashed_validation': True, 'budget': expected,
             'observed': {k: completion.get(k) for k in expected},
             'new_training_seeds': list(SEEDS), 'old_reference_seed': 23005,
             'old_reference_excluded_from_R_M': True,
             'input_file_hashes_checked': len(completion['data_files']),
             'artifact_hashes_checked': sum(len(r['artifact_sha256']) for r in completion['records']),
             'reviewed_seed_decisions': decision['seed_decisions'],
             'all_negative_deltas': {c: decision['methods'][c]['all_negative_deltas'] for c in METHODS[2:]},
             'discrepancies': discrepancies}
    review = {'decision': decision['decision'] if not discrepancies else 'INCONCLUSIVE',
              'substatus': decision['substatus'] if not discrepancies else 'REPRODUCTION_MISMATCH',
              'R': decision['R'], 'M': decision['M'], 'independent_decision': decision,
              'reproduction_passed': not discrepancies, 'gate1_claim': False,
              'strongest_alternative': 'Supervised finite-state codebook; calibration labels and prototype state can explain rescue without a unique topology mechanism.',
              'interpretation_limit': 'Conditional failure and M-eligible denominators are selected by fixed qualification/control criteria; no biological or Gate 1 conclusion.',
              'failure_seed_count': len(decision['qualified_failure_seeds']),
              'M_eligible_count': len(decision['M_eligible_seeds']),
              'reference_excluded': 23005}
    return reproduction, audit, review


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--frozen', type=Path, required=True)
    parser.add_argument('--report-dir', type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    report_dir = args.report_dir.resolve()
    if report_dir != Path(__file__).resolve().parent:
        raise ValueError('Reports must remain in ITER-0020')
    if any((report_dir / n).exists() for n in ('reproduction.json', 'audit_execution.json', 'review.json')):
        raise FileExistsError('Single pass only; existing review output must be preserved')
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('600-second replay alarm')))
    signal.alarm(CAP)
    try:
        reproduction, audit, review = execute(args.run.resolve(), args.frozen.resolve(), report_dir)
    except BaseException as exc:
        reproduction = {'status': 'FAILED', 'error': repr(exc), 'traceback': traceback.format_exc(),
                        'wall_seconds': time.monotonic() - START, 'cap_seconds': CAP,
                        'script_sha256': digest(Path(__file__)), 'single_attempt': True}
        audit = {'status': 'FAILED', 'error': repr(exc), 'formal_output': str(args.run.resolve())}
        review = {'decision': 'INCONCLUSIVE', 'substatus': 'INDEPENDENT_REPLAY_FAILED',
                  'reproduction_passed': False, 'error': repr(exc), 'gate1_claim': False}
    finally:
        signal.alarm(0)
    for filename, value in (('reproduction.json', reproduction),
                            ('audit_execution.json', audit), ('review.json', review)):
        with (report_dir / filename).open('x') as f:
            json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)
            f.write('\n')
    print(json.dumps({'reproduction': reproduction['status'], 'decision': review['decision'],
                      'wall_seconds': time.monotonic() - START}))


if __name__ == '__main__':
    main()
