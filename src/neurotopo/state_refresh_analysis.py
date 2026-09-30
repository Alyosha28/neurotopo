"""EXP0024v1 raw-array audit and fixed exposed-case decision."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from .order_observation import sha
from .run import source_fingerprints

SEEDS = tuple(range(23001, 23007))
ARMS = ('REL_BASE', 'INV_BASE')
CONDITIONS = ('NONE', 'GAUGE8', 'RESET8', 'RESET1')
CELLS = ('IID_H', 'IID_U', 'LONG_H', 'LONG_U')


def scores(logits: np.ndarray, target: np.ndarray, commands: np.ndarray) -> dict[str, Any]:
    z = np.asarray(logits, np.float64)
    if z.shape != target.shape or z.shape != (len(commands), commands.shape[1] + 1):
        raise ValueError('Wrong prediction, target or command shape')
    if not np.isfinite(z).all():
        raise FloatingPointError('Nonfinite logits')
    expected = np.bitwise_xor.accumulate(np.column_stack((target[:, 0].astype(np.int64), commands)), axis=1)
    if not np.array_equal(target, expected):
        raise RuntimeError('Evaluation targets violate independent XOR oracle')
    correct = (z >= 0) == target
    errors = ~correct
    ever = errors.any(axis=1)
    first = np.argmax(errors, axis=1)
    hold, flip = commands == 0, commands == 1
    loss = np.logaddexp(0., z) - z * target
    return {'bce': float(loss.mean()), 'future_bce': float(loss[:, 1:].mean()),
            'accuracy_all': float(correct.mean()), 'future_accuracy': float(correct[:, 1:].mean()),
            'final_accuracy': float(correct[:, -1].mean()),
            'initial_accuracy': float(correct[:, 0].mean()),
            'hold_accuracy': float(correct[:, 1:][hold].mean()) if hold.any() else None,
            'flip_accuracy': float(correct[:, 1:][flip].mean()) if flip.any() else None,
            'first_error_count': int(ever.sum()),
            'first_error_mean_time': float(first[ever].mean()) if ever.any() else None,
            'first_error_by_time': np.bincount(first[ever], minlength=z.shape[1]).tolist(),
            'accuracy_by_time': correct.mean(axis=0).tolist()}


def refresh_diagnostics(raw: dict[str, np.ndarray], condition: str) -> dict[str, Any]:
    mask = np.asarray(raw['refresh_mask'], bool)
    z, g, pz, pg = (raw[k] for k in ('logits', 'gram', 'post_logits', 'post_gram'))
    selected = raw['selected_labels']
    if condition not in CONDITIONS or z.shape != pz.shape or g.shape != pg.shape \
            or g.shape != (*z.shape, 3) or mask.shape != (z.shape[1],) \
            or selected.shape != z.shape or mask[0]:
        raise RuntimeError('Invalid refresh audit shapes')
    expected_mask = np.zeros_like(mask)
    if condition != 'NONE':
        expected_mask[1:] = np.arange(1, len(mask)) % (1 if condition == 'RESET1' else 8) == 0
    if not np.array_equal(mask, expected_mask):
        raise RuntimeError('Wrong registered refresh schedule')
    if not np.array_equal(z[:, ~mask], pz[:, ~mask]) or not np.array_equal(g[:, ~mask], pg[:, ~mask]):
        raise RuntimeError('Unscheduled state change')
    if np.any(selected[:, ~mask] != -1):
        raise RuntimeError('Selected label without refresh')
    if condition in ('NONE', 'GAUGE8'):
        if np.any(selected != -1):
            raise RuntimeError('Control selected a prototype')
    elif not np.array_equal(selected[:, mask], (z[:, mask] >= 0).astype(np.int64)):
        raise RuntimeError('Prototype selection did not follow own current logit sign')
    dz = np.abs(pz[:, mask].astype(np.float64) - z[:, mask].astype(np.float64))
    dg = np.abs(pg[:, mask].astype(np.float64) - g[:, mask].astype(np.float64))
    sign_changed = ((pz[:, mask] >= 0) != (z[:, mask] >= 0))
    result = {'refresh_events': int(mask.sum()), 'refresh_calls': int(mask.sum() * len(z)),
              'local_logit_abs_max': float(dz.max()) if dz.size else 0.,
              'local_gram_abs_max': float(dg.max()) if dg.size else 0.,
              'mean_gram_l2_change': float(np.linalg.norm((pg[:, mask] - g[:, mask]).astype(np.float64), axis=-1).mean()) if dg.size else 0.,
              'post_sign_change_rate': float(sign_changed.mean()) if sign_changed.size else 0.}
    if 'state_delta_norm' not in raw:
        raise RuntimeError('Missing registered state delta audit')
    delta = np.asarray(raw['state_delta_norm'])
    if delta.shape != z.shape or np.any(delta < 0) or not np.isfinite(delta).all() \
            or np.any(delta[:, ~mask] != 0):
        raise RuntimeError('Invalid state delta audit')
    result['mean_state_l2_change'] = float(delta[:, mask].mean()) if mask.any() else 0.
    result['max_state_l2_change'] = float(delta[:, mask].max()) if mask.any() else 0.
    return result


def summarize_six(values: list[float]) -> dict[str, Any]:
    x = np.asarray(values, np.float64)
    if x.shape != (6,) or not np.isfinite(x).all():
        raise ValueError('Six finite exposed seed effects required')
    mean, sd = float(x.mean()), float(x.std(ddof=1))
    half = 2.570581836 * sd / np.sqrt(6)
    return {'values_by_seed': {str(seed): float(v) for seed, v in zip(SEEDS, x)},
            'mean': mean, 'sample_sd': sd, 'min': float(x.min()), 'max': float(x.max()),
            'descriptive_t95': [mean - half, mean + half]}


def decide(rows: list[dict[str, Any]]) -> dict[str, Any]:
    lookup = {(r['seed'], r['arm']): r for r in rows}
    if len(rows) != 12 or set(lookup) != {(s, a) for s in SEEDS for a in ARMS}:
        raise ValueError('Incomplete 12-model matrix')
    case = lookup[23005, 'REL_BASE']
    def score(condition: str, cell: str) -> dict[str, Any]:
        return case['evaluation'][condition][cell]['metrics']
    def long_average(condition: str) -> float:
        return float(np.mean([score(condition, cell)['final_accuracy'] for cell in ('LONG_H', 'LONG_U')]))
    base, gauge, reset = (long_average(c) for c in ('NONE', 'GAUGE8', 'RESET8'))
    calibration = case['calibration']
    qualified = {
        'calibration_accuracy': calibration['calibration_accuracy'] >= .95,
        'prototype_labels': bool(calibration['prototype_labels_correct']),
        'none_iid_h_all': score('NONE', 'IID_H')['accuracy_all'] >= .95,
        'none_iid_u_all': score('NONE', 'IID_U')['accuracy_all'] >= .95,
        'none_iid_u_flip': score('NONE', 'IID_U')['flip_accuracy'] >= .95,
        'gauge_local': all(case['evaluation']['GAUGE8'][cell]['diagnostics']['local_logit_abs_max'] <= 1e-5
                           and case['evaluation']['GAUGE8'][cell]['diagnostics']['local_gram_abs_max'] <= 1e-5
                           for cell in CELLS),
        'gauge_long_equivalence': abs(base - gauge) <= .005,
    }
    if not all(qualified.values()):
        decision, sub = 'INCONCLUSIVE', 'CASE_QUALIFICATION_FAILURE'
    elif base < .95 and reset >= .95 and reset - gauge >= .10:
        decision, sub = 'SUPPORTED', 'FROZEN_CASE_STATE_REFRESH_RESCUE'
    elif reset - gauge >= .05:
        decision, sub = 'PARTIALLY_SUPPORTED', 'SMALLER_CASE_RESCUE'
    else:
        decision, sub = 'INCONCLUSIVE', 'NO_REGISTERED_CASE_RESCUE'
    effects = {}
    for comparator in ('NONE', 'GAUGE8'):
        effects[f'RESET8_minus_{comparator}'] = summarize_six([
            float(np.mean([lookup[s, 'REL_BASE']['evaluation']['RESET8'][cell]['metrics']['final_accuracy']
                           - lookup[s, 'REL_BASE']['evaluation'][comparator][cell]['metrics']['final_accuracy']
                           for cell in ('LONG_H', 'LONG_U')])) for s in SEEDS])
    return {'decision': decision, 'substatus': sub, 'case_seed': 23005,
            'case_qualification': qualified, 'case_long_final_accuracy': {'NONE': base, 'GAUGE8': gauge,
                'RESET8': reset, 'RESET1': long_average('RESET1')},
            'case_reset8_minus_gauge8': reset - gauge,
            'exposed_rel_descriptive_effects': effects}


def _same_numbers(a: dict[str, Any], b: dict[str, Any], tolerance: float = 1e-12) -> None:
    if set(a) != set(b):
        raise RuntimeError(f'Metric keys changed: {set(a) ^ set(b)}')
    for k in a:
        x, y = a[k], b[k]
        if isinstance(x, float) and isinstance(y, float):
            if not np.isclose(x, y, rtol=0, atol=tolerance):
                raise RuntimeError(f'Metric mismatch: {k}: {x} != {y}')
        elif x != y:
            raise RuntimeError(f'Metric mismatch: {k}')


def analyze(out: Path) -> dict[str, Any]:
    started = time.monotonic()
    failure = out / 'run_failure.json'
    worker_failure = out / 'worker_failure.json'
    if failure.exists() or worker_failure.exists():
        detail = _read(failure) if failure.exists() else {}
        worker = _read(worker_failure) if worker_failure.exists() else {}
        error_text = detail.get('error', '') + worker.get('error', '')
        result = {'experiment_id': 'EXP-0024', 'version': 'v1', 'decision': 'INCONCLUSIVE',
                  'substatus': 'INCOMPLETE_BUDGET' if 'TimeoutError' in error_text else 'ENGINEERING_ABORT',
                  'detail': detail, 'worker': worker, 'gate1_claim': False}
        _save_compact(out / 'analysis.json', result)
        return result
    manifest, complete = _read(out / 'manifest.json'), _read(out / 'completion.json')
    if manifest['source_sha256'] != source_fingerprints() or complete['status'] != 'COMPLETED':
        raise RuntimeError('Source or run completion mismatch')
    for field in ('frozen_hashes', 'parent_artifact_sha256'):
        for path, digest in manifest[field].items():
            if sha(Path(path)) != digest:
                raise RuntimeError(f'Frozen source changed: {path}')
    if sha(out / 'parent-manifest.json') != manifest['parent_artifact_sha256'][str(Path(manifest['config']['parent']) / 'manifest.json')]:
        raise RuntimeError('Copied parent manifest changed')
    if sha(out / 'parent-completion.json') != manifest['parent_artifact_sha256'][str(Path(manifest['config']['parent']) / 'completion.json')]:
        raise RuntimeError('Copied parent completion changed')
    for path, digest in complete['data_files'].items():
        if sha(out / path) != digest:
            raise RuntimeError(f'Input changed: {path}')
    rows = complete['records']
    if complete['research_updates'] != 0 or complete['prototype_fits'] != 12 \
            or complete['evaluation_count'] != 192 or complete['evaluation_transition_tokens'] != 12976128:
        raise RuntimeError('Run count violates registration')
    if len(rows) != 12 or {(r['seed'], r['arm']) for r in rows} != {(s, a) for s in SEEDS for a in ARMS}:
        raise RuntimeError('Model matrix incomplete')
    reporting = []
    for row in rows:
        seed, arm = row['seed'], row['arm']
        path = out / str(seed) / arm
        if _read(path / 'result.json') != row:
            raise RuntimeError('Row/completion disagreement')
        for name, digest in row['artifact_sha256'].items():
            if sha(path / name) != digest:
                raise RuntimeError(f'Model artifact changed: {seed}/{arm}/{name}')
        with np.load(path / 'prototypes.npz') as p, np.load(out / 'data' / str(seed) / 'calibration.npz') as cal:
            labels = cal['targets'][:, -1]
            if not np.array_equal(p['calibration_labels'], labels):
                raise RuntimeError('Calibration labels changed')
            if not np.array_equal((p['prototype_logits'] >= 0), np.array([False, True])) == row['calibration']['prototype_labels_correct']:
                raise RuntimeError('Prototype label summary mismatch')
            if not np.isclose(np.mean((p['calibration_logits'][:, -1] >= 0) == labels), row['calibration']['calibration_accuracy']):
                raise RuntimeError('Calibration accuracy mismatch')
            g = p['calibration_states'].astype(np.float64).transpose(0, 2, 1) @ p['calibration_states'].astype(np.float64)
            means = np.stack([g[labels == b].mean(0) for b in (0, 1)])
            if not np.allclose(means, p['prototype_grams'], rtol=0, atol=1e-12):
                raise RuntimeError('Prototype Gram fit mismatch')
            proto_g = p['prototypes'].astype(np.float64).transpose(0, 2, 1) @ p['prototypes'].astype(np.float64)
            if np.max(np.abs(proto_g - means)) > 1e-5:
                raise RuntimeError('Prototype Gram reconstruction mismatch')
        with np.load(path / 'transition_table.npz') as table:
            if not np.array_equal(table['predictions'], table['logits'] >= 0) \
                    or not np.array_equal(table['expected'], np.array([[0, 1], [1, 0]])):
                raise RuntimeError('Static transition table changed')
            transition_correct = bool(np.array_equal(table['predictions'], table['expected']))
            transition_predictions = table['predictions'].tolist()
        compact = {'seed': seed, 'arm': arm, 'calibration': row['calibration'],
                   'static_transition_correct': transition_correct,
                   'static_transition_table': transition_predictions,
                   'resources': row['resources'], 'evaluation': {}}
        for condition in CONDITIONS:
            compact['evaluation'][condition] = {}
            for cell in CELLS:
                record = row['evaluation'][condition][cell]
                raw_path = path / condition / f'{cell}.npz'
                if sha(raw_path) != record['artifact_sha256']:
                    raise RuntimeError(f'Raw rollout changed: {seed}/{arm}/{condition}/{cell}')
                with np.load(raw_path) as raw, np.load(out / 'data' / str(seed) / f'{cell}.npz') as data:
                    checked = scores(raw['logits'], data['targets'], data['commands'])
                    diagnostic = refresh_diagnostics({k: raw[k] for k in raw.files}, condition)
                _same_numbers(checked, record['metrics'])
                _same_numbers(diagnostic, record['diagnostics'])
                compact['evaluation'][condition][cell] = {
                    'metrics': {k: v for k, v in checked.items()
                                if k not in ('accuracy_by_time', 'first_error_by_time')},
                    'diagnostics': diagnostic, 'raw_path': str(raw_path.relative_to(out))}
        reporting.append(compact)
    verdict = decide(rows)
    result = {'experiment_id': 'EXP-0024', 'version': 'v1', **verdict,
              'models': reporting, 'model_count': 12, 'prototype_fits': 12,
              'evaluation_count': 192, 'research_updates': 0, 'gate1_claim': False,
              'matching_criterion_version': 'G1-MATCH-v3',
              'source_sha256': manifest['source_sha256'],
              'manifest_sha256': sha(out / 'manifest.json'),
              'completion_sha256': sha(out / 'completion.json'),
              'scope': 'Prespecified exposed REL23005 frozen case; other exposed seeds descriptive; supervised two-state codebook.',
              'run_seconds': complete['wall_seconds'], 'analysis_seconds': time.monotonic() - started}
    if result['run_seconds'] + result['analysis_seconds'] > 900:
        raise TimeoutError('EXP0024 run and analysis exceeded registered 900 seconds')
    _save_compact(out / 'analysis.json', result)
    if (out / 'analysis.json').stat().st_size > 250000:
        raise RuntimeError('Compact analysis exceeds reflection packet size')
    return result


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _save_compact(path: Path, data: dict[str, Any]) -> None:
    encoded = json.dumps(data, separators=(',', ':'), ensure_ascii=False, allow_nan=False) + '\n'
    with path.open('x') as stream:
        stream.write(encoded)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    answer = analyze(args.run)
    print(json.dumps({k: answer[k] for k in ('experiment_id', 'decision', 'substatus')}))
