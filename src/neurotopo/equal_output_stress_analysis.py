"""Prespecified EXP-0022 v2 reductions from immutable cell trajectories."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from .equal_output_stress import ARMS, CONDITIONS, RADII, SEEDS
from .frozen_binary_probe import save_json
from .order_observation import sha
from .run import source_fingerprints

T95 = 3.182446305


def _require(ok: bool, why: str) -> None:
    if not ok:
        raise ValueError(why)


def _bce(logits: np.ndarray, targets: np.ndarray) -> np.ndarray:
    return np.logaddexp(0., logits.astype(np.float64)) - logits.astype(np.float64) * targets


def _summary(values: dict[int, float]) -> dict[str, Any]:
    _require(set(values) == set(SEEDS[:4]), 'Four prior confirmation seed values required')
    x = np.asarray([values[seed] for seed in SEEDS[:4]], dtype=np.float64)
    _require(np.isfinite(x).all(), 'Nonfinite seed contrast')
    mean, sd = float(x.mean()), float(x.std(ddof=1))
    return {'by_seed': {str(seed): float(values[seed]) for seed in SEEDS[:4]},
            'mean': mean, 'sample_sd': sd, 'min': float(x.min()),
            'max': float(x.max()), 'descriptive_t95': [mean - T95 * sd / 2,
                                                       mean + T95 * sd / 2]}


def _condition(logits: np.ndarray, targets: np.ndarray) -> tuple[dict[str, Any], dict[str, Any]]:
    loss = _bce(logits, targets)
    correct = (logits >= 0) == (targets >= .5)
    future = correct[..., 1:]
    # A trajectory without an error is right censored at 32. The explicit
    # no-error count disambiguates it from a first error at step 32.
    any_error = (~future).any(axis=-1)
    first = np.where(any_error, np.argmax(~future, axis=-1) + 1, 32)
    hist = np.bincount(first[any_error].ravel(), minlength=33).astype(int)
    metrics = {'current_accuracy': float(correct[..., 0].mean()),
               'future_bce': float(loss[..., 1:].mean()),
               'future_accuracy': float(future.mean()),
               'last_accuracy': float(correct[..., -1].mean()),
               'first_error_counts_1_to_32': hist[1:].tolist(),
               'right_censored_no_error_count': int((~any_error).sum()),
               'trajectories': int(any_error.size)}
    curves = {'bce_by_step_0_to_32': loss.mean(axis=(0, 1)).tolist(),
              'accuracy_by_step_0_to_32': correct.mean(axis=(0, 1)).tolist()}
    return metrics, curves


def _cell(raw: Any, targets: np.ndarray, seed: int, arm: str, radius: float
          ) -> tuple[dict[str, Any], dict[str, Any]]:
    logits = raw['test_logits']
    grams = raw['test_gram']
    _require(logits.shape == (64, 4, 16, 33) and grams.shape == (64, 4, 16, 33, 3),
             'Wrong test trajectory shape')
    _require(np.isfinite(logits).all() and np.isfinite(grams).all(), 'Nonfinite test trajectory')
    _require(raw['candidate_states'].shape == (64, 16, 3, 2)
             and raw['condition_states'].shape == (64, 4, 3, 2), 'Wrong state bank shape')
    _require(raw['selection_scores'].shape == (64, 16)
             and raw['selection_indices'].shape == (64,), 'Wrong selection shape')
    _require(raw['condition_valid'][:, 0].all(), 'NONE cannot be invalid')
    _require(np.array_equal(raw['condition_valid'][:, 1], raw['valid'][:, 0])
             and np.array_equal(raw['condition_valid'][:, 2],
                                raw['valid'][np.arange(64), raw['selection_indices']])
             and np.array_equal(raw['condition_valid'][:, 3], raw['sham_valid']),
             'Invalid fallback flags')
    # Invalid prefix states stay in every descriptive denominator and revert
    # exactly to NONE at t0; all candidate and sham validity defines the guard.
    fallback = ~raw['condition_valid'].astype(bool)
    for condition in range(1, 4):
        _require(np.array_equal(raw['condition_states'][fallback[:, condition], condition],
                                raw['condition_states'][fallback[:, condition], 0]),
                 'Invalid prefix did not fall back to NONE')
    score: dict[str, Any] = {}
    curves: dict[str, Any] = {}
    for i, name in enumerate(CONDITIONS):
        score[name], curves[name] = _condition(logits[:, i], targets)
    none, selected, random = score['NONE'], score['SELECTED'], score['RANDOM']
    all_valid = raw['valid'].astype(bool).all(axis=1) & raw['sham_valid'].astype(bool)
    logit_error = raw['logit_error']
    radius_error = raw['radius_error']
    norm_error = raw['norm_error']
    _require(logit_error.shape == radius_error.shape == norm_error.shape == (64, 16),
             'Wrong candidate constraint error shape')
    _require(np.isfinite(logit_error).all() and np.isfinite(radius_error).all()
             and np.isfinite(norm_error).all(), 'Nonfinite geometry constraint error')
    for key in ('sham_logit_error', 'sham_radius_error', 'sham_norm_error'):
        _require(raw[key].shape == (64,) and np.isfinite(raw[key]).all(),
                 f'Wrong or nonfinite {key}')
    search_selected = raw['selection_scores'][np.arange(64), raw['selection_indices']]
    search_random = raw['selection_scores'][:, 0]
    _require(np.isfinite(search_selected).all() and np.isfinite(search_random).all(),
             'Nonfinite effective search scores')
    score.update({'seed': seed, 'arm': arm, 'radius': radius,
                  'origin': 'EXPLORATORY_PILOT' if seed == 20001 else 'PRIOR_CONFIRMATION_SEED',
                  'all_candidate_and_sham_valid_fraction': float(all_valid.mean()),
                  'valid_candidate_fraction': float(raw['valid'].mean()),
                  'valid_sham_fraction': float(raw['sham_valid'].mean()),
                  'invalid_prefix_count': int((~all_valid).sum()),
                  'max_abs_current_logit_change': float(np.abs(logits[:, 1:, :, 0] - logits[:, :1, :, 0]).max()),
                  'max_candidate_logit_error': float(np.abs(logit_error).max()),
                  'max_candidate_relative_radius_error': float(np.abs(radius_error).max()),
                  'max_candidate_relative_norm_error': float(np.abs(norm_error).max()),
                  'max_sham_logit_error': float(np.abs(raw['sham_logit_error']).max()),
                  'max_sham_relative_radius_error': float(np.abs(raw['sham_radius_error']).max()),
                  'max_sham_relative_norm_error': float(np.abs(raw['sham_norm_error']).max()),
                  'direction_fallback_count': int(np.asarray(raw['direction_fallback']).sum()),
                  'sham_direction_fallback_count': int(np.asarray(raw['sham_direction_fallback']).sum()),
                  'selected_minus_none_future_bce': selected['future_bce'] - none['future_bce'],
                  'selected_minus_random_future_bce': selected['future_bce'] - random['future_bce'],
                  'search_selected_minus_random_bce': float(np.mean(search_selected - search_random)),
                  'test_last_accuracy_decline': none['last_accuracy'] - selected['last_accuracy'],
                  'sham_minus_none_future_bce': score['SHAM']['future_bce'] - none['future_bce'],
                  'sham_equivariance_alarm': bool(arm in ('INV', 'REL') and
                                                   abs(score['SHAM']['future_bce'] - none['future_bce']) > 1e-4)})
    return score, curves


def _decision(rows: list[dict[str, Any]]) -> dict[str, Any]:
    lookup = {(r['seed'], r['arm'], r['radius']): r for r in rows}
    primary = [lookup[seed, 'INV', .15] for seed in SEEDS[:4]]
    guards = all(r['all_candidate_and_sham_valid_fraction'] >= .99
                 and r['NONE']['current_accuracy'] >= .95
                 and r['NONE']['future_accuracy'] >= .95
                 for r in primary)
    large = [r['selected_minus_none_future_bce'] >= .05
             and r['test_last_accuracy_decline'] >= .05 for r in primary]
    small_bce = sum(r['selected_minus_none_future_bce'] >= .02 for r in primary)
    small_last = sum(r['test_last_accuracy_decline'] >= .02 for r in primary)
    if not guards:
        verdict, substatus = 'INCONCLUSIVE', 'QUALIFICATION_FAILURE'
    elif sum(large) >= 3:
        verdict, substatus = 'SUPPORTED', 'EQUAL_OUTPUT_FUTURE_VULNERABILITY'
    elif small_bce >= 3 and small_last >= 2:
        verdict, substatus = 'PARTIALLY_SUPPORTED', 'SMALLER_FUTURE_EFFECT'
    else:
        verdict, substatus = 'INCONCLUSIVE', 'NO_REGISTERED_WITNESS_FOUND'
    direction = [r['selected_minus_random_future_bce'] for r in primary]
    s_pass = float(np.mean(direction)) >= .02 and sum(x > 0 for x in direction) >= 3
    first_radius = {}
    for seed in SEEDS:
        first_radius[str(seed)] = {}
        for arm in ARMS:
            observed = [radius for radius in RADII
                        if lookup[seed, arm, radius]['test_last_accuracy_decline'] >= .05]
            first_radius[str(seed)][arm] = min(observed) if observed else None
    return {'decision': verdict, 'substatus': substatus, 'E_guards_pass': guards,
            'E_large_seed_count': int(sum(large)),
            'E_small_bce_seed_count': int(small_bce),
            'E_small_last_seed_count': int(small_last),
            'S_directional_search_pass': bool(s_pass),
            'primary_selected_minus_none_future_bce': _summary({r['seed']: r['selected_minus_none_future_bce'] for r in primary}),
            'primary_last_accuracy_decline': _summary({r['seed']: r['test_last_accuracy_decline'] for r in primary}),
            'secondary_selected_minus_random_future_bce': _summary({r['seed']: r['selected_minus_random_future_bce'] for r in primary}),
            'first_observed_grid_radius_for_last_accuracy_decline_005': first_radius,
            'sham_equivariance_alarm_cells': [f"{r['seed']}/{r['arm']}/{r['radius']}" for r in rows
                                               if r['sham_equivariance_alarm']]}


def analyze(out: Path) -> dict[str, Any]:
    start = time.monotonic()
    failure = out / 'worker_failure.json'
    run_failure = out / 'run_failure.json'
    if failure.exists() or run_failure.exists():
        detail = json.loads(run_failure.read_text()) if run_failure.exists() else {}
        worker = json.loads(failure.read_text()) if failure.exists() else {}
        sub = ('INCOMPLETE_BUDGET' if 'TimeoutError' in detail.get('error', '')
               else worker.get('substatus', 'ENGINEERING_ABORT'))
        result = {'experiment_id': 'EXP-0022', 'version': 'v2', 'decision': 'INCONCLUSIVE',
                  'substatus': sub, 'failure': worker, 'run_failure': detail,
                  'gate1_claim': False, 'research_updates': 0}
        save_json(out / 'analysis.json', result)
        return result
    manifest = json.loads((out / 'manifest.json').read_text())
    completed = json.loads((out / 'completion.json').read_text())
    _require(manifest['source_sha256'] == source_fingerprints(), 'Source changed since run')
    for name, digest in {**manifest['frozen_hashes'], **manifest['parent_artifact_sha256']}.items():
        _require(sha(Path(name)) == digest, f'Frozen artifact changed: {name}')
    _require(completed['status'] == 'COMPLETED' and completed['research_updates'] == 0
             and completed['optimizer_updates'] == 0, 'Incomplete or updated frozen run')
    _require(completed['candidate_states'] == 46080
             and completed['search_suffix_rollouts'] == 184320
             and completed['test_suffix_rollouts'] == 184320,
             'Wrong executed workload')
    for name, digest in completed['data_files'].items():
        _require(sha(out / name) == digest, f'Data hash mismatch: {name}')
    _require(len(completed['data_files']) == 15, 'Expected three shared datasets per seed')
    records = completed['records']
    expected = {(seed, arm, radius) for seed in SEEDS for arm in ARMS for radius in RADII}
    _require(len(records) == 45 and {(r['seed'], r['arm'], r['radius']) for r in records} == expected,
             'Expected complete 45 cell matrix')
    scores: list[dict[str, Any]] = []
    curves: dict[str, Any] = {}
    test_targets: dict[int, np.ndarray] = {}
    for record in records:
        seed, arm, radius = record['seed'], record['arm'], record['radius']
        cell_dir = out / record['path']
        for name, digest in record['artifact_sha256'].items():
            _require(sha(cell_dir / name) == digest,
                     f'Cell artifact hash mismatch: {seed}/{arm}/{radius}/{name}')
        saved_record = json.loads((cell_dir / 'record.json').read_text())
        expected_record = {**record, 'artifact_sha256':
                           {key: value for key, value in record['artifact_sha256'].items()
                            if key != 'record.json'}}
        _require(saved_record == expected_record, 'Cell record disagrees with completion')
        _require(sha(cell_dir.parent / 'checkpoint.pt') == record['parent_checkpoint_sha256'],
                 'Cell copied checkpoint changed')
        _require(record['model_state_hash_before'] == record['model_state_hash_after'],
                 'Frozen model state changed')
        if seed not in test_targets:
            with np.load(out / str(seed) / 'data' / 'test.npz') as data:
                test_targets[seed] = data['targets']
        _require(test_targets[seed].shape == (64, 16, 33), 'Wrong held-out target shape')
        with np.load(cell_dir / 'raw.npz') as raw, np.load(cell_dir / 'selection.npz') as selection:
            _require(np.array_equal(raw['selection_indices'], selection['indices'])
                     and np.array_equal(raw['selection_scores'], selection['scores']),
                     'Selection receipt disagrees with raw record')
            score, curve = _cell(raw, test_targets[seed], seed, arm, radius)
        score['resources'] = record['resources']
        score['search_wall_seconds'] = record['search_wall_seconds']
        score['search_cpu_seconds'] = record['search_cpu_seconds']
        score['test_write_wall_seconds'] = record['test_write_wall_seconds']
        score['test_write_cpu_seconds'] = record['test_write_cpu_seconds']
        scores.append(score)
        curves[f'{seed}/{arm}/r{round(radius * 1000):03d}'] = curve
    decision = _decision(scores)
    by_cell = {(r['seed'], r['arm'], r['radius']): r for r in scores}
    for score in scores:
        raw = by_cell[score['seed'], 'RAW', score['radius']]
        score['resource_ratios_vs_RAW_same_seed_radius'] = {
            'parameter_bytes': score['resources']['parameter_bytes'] / raw['resources']['parameter_bytes'],
            'state_bytes_per_example': score['resources']['state_bytes_per_example'] / raw['resources']['state_bytes_per_example'],
            'measured_search_and_test_cpu_seconds':
                (score['search_cpu_seconds'] + score['test_write_cpu_seconds']) /
                (raw['search_cpu_seconds'] + raw['test_write_cpu_seconds']),
            'measured_search_and_test_wall_seconds':
                (score['search_wall_seconds'] + score['test_write_wall_seconds']) /
                (raw['search_wall_seconds'] + raw['test_write_wall_seconds'])}
    save_json(out / 'curves.json', curves)
    result = {'experiment_id': 'EXP-0022', 'version': 'v2',
              'matching_criterion_version': 'G1-MATCH-v3', 'gate1_claim': False,
              'origin_note': '20001 exploratory; 20002..20005 reused trained seeds with new inputs and suffixes',
              'primary_arm': 'INV', 'primary_radius': .15,
              **decision, 'cells': scores, 'curves_sha256': sha(out / 'curves.json'),
              'source_sha256': manifest['source_sha256'],
              'config_sha256': manifest['config_sha256'],
              'manifest_sha256': sha(out / 'manifest.json'),
              'completion_sha256': sha(out / 'completion.json'),
              'run_wall_seconds': completed['wall_seconds'],
              'run_cpu_seconds': completed['cpu_seconds'],
              'analysis_wall_seconds': time.monotonic() - start,
              'research_updates': 0,
              'efficiency_claim': False}
    _require(completed['wall_seconds'] + result['analysis_wall_seconds'] <= 900,
             'Shared run and analysis wall budget exceeded')
    save_json(out / 'analysis.json', result)
    _require((out / 'analysis.json').stat().st_size < 250_000,
             'Compact analysis size exceeds 250KB')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.run)
    print(json.dumps({k: result[k] for k in ('experiment_id', 'decision', 'substatus')},
                     ensure_ascii=False), flush=True)
