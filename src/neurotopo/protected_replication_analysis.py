"""Registered EXP-0021 v1 decisions from complete frozen artifacts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from .frozen_binary_probe import save_json
from .learned_protected_pilot import ARMS, CELLS, learning_flag
from .order_observation import sha
from .protected_replication import CONDITIONS, MONITOR_STEPS, PARAMETERS, PILOT_SEED, SEEDS
from .run import source_fingerprints


T_CRITICAL = 3.182446305
CONTRAST_THRESHOLD = .10
QUALIFICATION_THRESHOLD = .95
ROTATION_GAP = .20


def _expected_tokens(step: int) -> int:
    return 64 * (min(step, 125)
                 + 2 * max(0, min(step, 250) - 125)
                 + 4 * max(0, min(step, 500) - 250)
                 + 8 * max(0, min(step, 1000) - 500))


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _summarize(values: list[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    _require(array.shape == (4,) and np.isfinite(array).all(), 'Four finite new-seed values required')
    mean = float(array.mean())
    sd = float(array.std(ddof=1))
    half = T_CRITICAL * sd / np.sqrt(4)
    return {'values_by_seed': {str(seed): float(value) for seed, value in zip(SEEDS, array)},
            'mean': mean, 'sample_sd': sd, 'min': float(array.min()),
            'max': float(array.max()), 'descriptive_t95': [mean - half, mean + half]}


def _pilot_compare(actual: dict[str, Any], previous: dict[str, Any]) -> None:
    for cell in CELLS:
        a, b = actual[cell], previous[cell]
        _require(set(a) == set(b), f'Pilot {cell} score schema changed')
        for key, value in a.items():
            old = b[key]
            if value is None:
                _require(old is None, f'Pilot {cell}/{key} missingness changed')
            elif isinstance(value, list):
                np.testing.assert_allclose(value, old, atol=1e-8, rtol=0)
            elif isinstance(value, (int, float)):
                _require(abs(value - old) <= 1e-8, f'Pilot {cell}/{key} replay mismatch')
            else:
                _require(value == old, f'Pilot {cell}/{key} replay mismatch')


def _check_artifacts(out: Path, manifest: dict[str, Any], completed: dict[str, Any]) -> list[dict[str, Any]]:
    _require(manifest['source_sha256'] == source_fingerprints(), 'Source differs from frozen run')
    for path, digest in manifest['frozen_hashes'].items():
        _require(sha(Path(path)) == digest, f'Frozen file changed: {path}')
    for path, digest in manifest['pilot_artifact_sha256'].items():
        _require(sha(Path(path)) == digest, f'Exploratory artifact changed: {path}')
    _require(completed['status'] == 'COMPLETED' and completed['updates'] == 12000,
             'Incomplete research update count')
    for name, digest in completed['data_files'].items():
        _require(sha(out / name) == digest, f'Dataset hash mismatch: {name}')
    _require(len(completed['data_files']) == 50, 'Expected ten datasets per seed')
    rows = completed['records']
    expected = {(seed, arm) for seed in (*SEEDS, PILOT_SEED) for arm in ARMS}
    _require(len(rows) == 15 and {(r['seed'], r['arm']) for r in rows} == expected,
             'Expected fifteen unique models')
    for row in rows:
        seed, arm = row['seed'], row['arm']
        path = out / str(seed) / arm
        full = json.loads((path / 'result.json').read_text())
        _require({k: v for k, v in full.items() if k != 'losses'} == row,
                 f'Summary/result mismatch: {seed}/{arm}')
        _require(set(row['evaluation']) == set(CELLS), f'Missing evaluation cell: {seed}/{arm}')
        _require(set(row['intervention']) == set(CONDITIONS),
                 f'Missing intervention condition: {seed}/{arm}')
        for name, digest in row['artifact_sha256'].items():
            _require(sha(path / name) == digest, f'Output hash mismatch: {seed}/{arm}/{name}')
        _require(row['resources']['parameters'] == PARAMETERS[arm]
                 and row['resources']['parameter_bytes'] == 4 * PARAMETERS[arm]
                 and row['resources']['state_values'] == 6
                 and row['resources']['state_bytes_per_example'] == 24,
                 f'Frozen resource profile changed: {seed}/{arm}')
        if seed in SEEDS:
            _require(row['origin'] == 'CONFIRMATION' and row['updates'] == 1000
                     and len(full['losses']) == 1000, f'New-seed updates incomplete: {seed}/{arm}')
            _require(tuple(c['step'] for c in row['curves']) == MONITOR_STEPS,
                     f'Monitor checkpoint schedule changed: {seed}/{arm}')
            for curve in row['curves']:
                step = curve['step']
                _require(sha(path / f'checkpoint-{step}.pt') == curve['checkpoint_sha256'],
                         f'Checkpoint hash mismatch: {seed}/{arm}/{step}')
                _require(sha(path / f'monitor-{step}.npz') == curve['prediction_sha256'],
                         f'Monitor hash mismatch: {seed}/{arm}/{step}')
                _require(curve['resources']['updates'] == step and
                         curve['resources']['examples'] == step * 64 and
                         curve['resources']['operation_tokens'] == _expected_tokens(step),
                         f'Monitor resource count mismatch: {seed}/{arm}/{step}')
        else:
            _require(row['origin'] == 'EXPLORATORY_PILOT' and row['updates'] == 0
                     and row['source_training_updates'] == 1000 and row['curves'] == [],
                     f'Exploratory seed was retrained: {arm}')
            _require(row['resources']['examples'] == row['resources']['operation_tokens']
                     == row['resources']['updates'] == 0,
                     f'Pilot import charged as new training: {arm}')
            _pilot_compare(row['evaluation'], row['source_evaluation'])
    for seed in (*SEEDS, PILOT_SEED):
        trio = [r for r in rows if r['seed'] == seed]
        history = 'training_data_sha256' if seed in SEEDS else 'source_training_data_sha256'
        init = 'initial_hashes' if seed in SEEDS else 'source_initial_hashes'
        _require(len({r[history] for r in trio}) == 1, f'Unequal training data: {seed}')
        for component in ('encoder', 'readout'):
            _require(len({r[init][component] for r in trio}) == 1,
                     f'Unequal initial {component}: {seed}')
    _require(len({next(r for r in rows if r['seed'] == seed and r['arm'] == 'RAW')
                  ['training_data_sha256'] for seed in SEEDS}) == 4,
             'New seeds have identical training data')
    return rows


def _seed_table(rows: list[dict[str, Any]], seed: int) -> dict[str, Any]:
    by_arm = {r['arm']: r for r in rows if r['seed'] == seed}
    _require(set(by_arm) == set(ARMS), f'Incomplete seed {seed}')
    def acc(arm: str, condition: str) -> float:
        return by_arm[arm]['intervention'][condition]['immediate_accuracy']

    d_norm = {arm: acc(arm, 'NORM_SHAM') - acc(arm, 'NORM') for arm in ARMS}
    d_angle = {arm: acc(arm, 'ANGLE_SHAM') - acc(arm, 'ANGLE') for arm in ARMS}
    c_norm = d_norm['INV'] - d_norm['REL']
    c_angle = d_angle['REL'] - d_angle['INV']
    learned = {arm: learning_flag(by_arm[arm]) for arm in ARMS}
    rot = {arm: by_arm[arm]['evaluation']['ROT_U']['final_accuracy'] for arm in ARMS}
    gap = {arm: rot[arm] - rot['RAW'] for arm in ('INV', 'REL')}
    none = {arm: acc(arm, 'NONE') for arm in ARMS}
    replicate = (all(learned[arm] for arm in ('INV', 'REL'))
                 and all(rot[arm] >= QUALIFICATION_THRESHOLD and gap[arm] >= ROTATION_GAP
                         for arm in ('INV', 'REL')))
    mechanism = (c_norm >= CONTRAST_THRESHOLD and c_angle >= CONTRAST_THRESHOLD
                 and all(none[arm] >= QUALIFICATION_THRESHOLD for arm in ('INV', 'REL')))
    curves = {}
    for arm, row in by_arm.items():
        observed = [curve['step'] for curve in row['curves']
                    if curve['metrics']['accuracy_all'] >= QUALIFICATION_THRESHOLD]
        curves[arm] = {'first_observed_monitor_accuracy_95_step': min(observed) if observed else None,
                       'monitor': row['curves']}
    return {'seed': seed, 'origin': 'CONFIRMATION' if seed in SEEDS else 'EXPLORATORY_PILOT',
            'iid_learnability': learned, 'rot_u_final_accuracy': rot, 'rot_u_raw_gap': gap,
            'none_immediate_accuracy': none, 'd_norm': d_norm, 'd_angle': d_angle,
            'c_norm': c_norm, 'c_angle': c_angle,
            'replication_pass': replicate, 'mechanism_pass': mechanism,
            'evaluation': {arm: by_arm[arm]['evaluation'] for arm in ARMS},
            'intervention': {arm: by_arm[arm]['intervention'] for arm in ARMS},
            'curves': curves}


def _decision(tables: list[dict[str, Any]]) -> tuple[str, str, bool, bool]:
    fresh = [t for t in tables if t['seed'] in SEEDS]
    _require(len(fresh) == 4, 'Four new seeds required for decision')
    r = all(t['replication_pass'] for t in fresh)
    m = all(t['mechanism_pass'] for t in fresh)
    if r and m:
        return 'SUPPORTED', 'REPLICATION_AND_CROSSOVER', r, m
    if r:
        return 'PARTIALLY_SUPPORTED', 'REPLICATION_ONLY', r, m
    if m:
        return 'PARTIALLY_SUPPORTED', 'CROSSOVER_ONLY', r, m
    qualified = all(all(t['iid_learnability'][arm] and
                        t['none_immediate_accuracy'][arm] >= QUALIFICATION_THRESHOLD
                        for arm in ('INV', 'REL')) for t in fresh)
    if qualified:
        return 'FALSIFIED', 'REGISTERED_JOINT_PREDICTION', r, m
    return 'INCONCLUSIVE', 'LEARNING_OR_ENGINEERING_UNRESOLVED', r, m


def analyze(out: Path) -> dict[str, Any]:
    failure_file = out / 'worker_failure.json'
    if failure_file.exists():
        failure = json.loads(failure_file.read_text())
        detail_file = out / 'run_failure.json'
        detail = json.loads(detail_file.read_text()) if detail_file.exists() else {}
        substatus = ('INCOMPLETE_BUDGET' if detail.get('error', '').startswith('TimeoutError(')
                     else failure['substatus'])
        result = {'experiment_id': 'EXP-0021', 'version': 'v1',
                  'decision': 'INCONCLUSIVE', 'substatus': substatus,
                  'failure': failure, 'detail': detail, 'gate1_claim': False}
    else:
        manifest = json.loads((out / 'manifest.json').read_text())
        completed = json.loads((out / 'completion.json').read_text())
        rows = _check_artifacts(out, manifest, completed)
        tables = [_seed_table(rows, seed) for seed in (*SEEDS, PILOT_SEED)]
        verdict, substatus, replication, mechanism = _decision(tables)
        fresh = tables[:4]
        stats: dict[str, dict[str, Any]] = {}
        for arm in ARMS:
            for name, extractor in (
                ('iid_h_accuracy', lambda t, a=arm: t['evaluation'][a]['IID_H']['accuracy_all']),
                ('iid_u_accuracy', lambda t, a=arm: t['evaluation'][a]['IID_U']['accuracy_all']),
                ('rot_u_final_accuracy', lambda t, a=arm: t['rot_u_final_accuracy'][a]),
                ('none_immediate_accuracy', lambda t, a=arm: t['none_immediate_accuracy'][a]),
                ('d_norm', lambda t, a=arm: t['d_norm'][a]),
                ('d_angle', lambda t, a=arm: t['d_angle'][a]),
            ):
                stats[f'{arm}_{name}'] = _summarize([extractor(t) for t in fresh])
        for name in ('c_norm', 'c_angle'):
            stats[name] = _summarize([t[name] for t in fresh])
        for arm in ('INV', 'REL'):
            stats[f'{arm}_rot_u_raw_gap'] = _summarize([t['rot_u_raw_gap'][arm] for t in fresh])
        result = {'experiment_id': 'EXP-0021', 'version': 'v1',
                  'decision': verdict, 'substatus': substatus,
                  'replication_R': replication, 'mechanism_M': mechanism,
                  'confirmation_seeds': list(SEEDS), 'exploratory_seed': PILOT_SEED,
                  'new_seed_tables': fresh, 'exploratory_pilot_table': tables[4],
                  'new_seed_descriptive_statistics': stats,
                  'five_seed_descriptive_only': {'seeds': [*SEEDS, PILOT_SEED],
                      'c_norm_mean': float(np.mean([t['c_norm'] for t in tables])),
                      'c_angle_mean': float(np.mean([t['c_angle'] for t in tables]))},
                  'resource_profile': {
                      arm: {'parameters': PARAMETERS[arm],
                            'parameter_bytes': 4 * PARAMETERS[arm],
                            'parameter_ratio_to_RAW': PARAMETERS[arm] / PARAMETERS['RAW'],
                            'state_values': 6, 'state_bytes_per_example': 24,
                            'executed_flops': 'NOT_MEASURED',
                            'peak_memory': 'NOT_MEASURED'} for arm in ARMS},
                  'run_seconds': completed['seconds'],
                  'run_cpu_seconds': completed['cpu_seconds'],
                  'updates': completed['updates'], 'gate1_claim': False,
                  'interpretation': ('Seed is the replication unit. Four new seeds alone determine R and M; '
                                     'the imported seed and 95% t intervals are descriptive. State interventions '
                                     'show frozen-checkpoint sensitivity, not training causality or Gate 1 evidence.')}
    save_json(out / 'analysis.json', result)
    print(json.dumps({k: v for k, v in result.items()
                      if k not in ('new_seed_tables', 'exploratory_pilot_table',
                                   'new_seed_descriptive_statistics')}), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    analyze(parser.parse_args().run)
