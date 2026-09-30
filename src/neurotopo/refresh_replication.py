"""EXP0025v2 fixed training, calibration and evaluation runner.

Scientific decisions are made by refresh_replication_analysis, never by this worker.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import signal
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

from . import refresh_factor_core as core
from .composition_core import predict_full
from .frozen_binary_probe import save_json
from .learned_protected_pilot import LearnedMemory, data_hash, generate, length_at, state_hash, tensors
from .order_observation import sha
from .run import git_output, source_fingerprints
from .state_refresh_core import fit_prototypes

SEEDS = tuple(range(25001, 25013))
REFERENCE_SEED = 23005
ARMS = ('REL', 'INV')
CONDITIONS = ('NONE', 'GAUGE8', 'FULL8', 'EQUAL8', 'SCALE8')
MONITOR_STEPS = (0, 125, 250, 500, 1000)
CELLS = {'IID_H': (8, 'HOLD', .4, 1), 'IID_U': (8, 'UPDATE', .4, 2),
         'LONG_H': (256, 'HOLD', np.pi, 1), 'LONG_U': (256, 'UPDATE', np.pi, 2)}
CAP = 1800
PARAMETERS = {'REL': 340, 'INV': 406}
TRAINING_TOKENS = 8_256_000
EVALUATION_TOKENS = 42_172_416


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        np.savez_compressed(stream, **arrays)


def _copy(source: Path, dest: Path, expected: str) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open('xb') as target, source.open('rb') as original:
        shutil.copyfileobj(original, target)
    if sha(dest) != expected:
        raise RuntimeError(f'Frozen copy hash mismatch: {source}')


def _budget(started: float) -> None:
    if time.monotonic() - started >= CAP:
        raise TimeoutError('EXP0025 run and analysis 1800-second budget exhausted')


def validate_config(c: dict[str, Any]) -> None:
    expected = {'experiment_id': 'EXP-0025', 'version': 'v2',
                'workflow': 'refresh_replication', 'protocol': 'experiments/EXP-0025/v2.md',
                'entry_contract': 'research/entries/EXP-0025.json',
                'output': 'artifacts/EXP-0025/v2', 'seeds': list(SEEDS),
                'arms': ['REL', 'INV'], 'conditions': list(CONDITIONS),
                'updates': 1000, 'batch_size': 64, 'examples': 512,
                'monitor_steps': list(MONITOR_STEPS), 'gate1_claim': False,
                'matching_criterion_version': 'G1-MATCH-v3',
                'resource': {'worker_wall_seconds_cap': CAP,
                             'reproduction_wall_seconds_cap': 600,
                             'research_update_cap': 24000,
                             'general_validation_overfit_cap': 900},
                'reference_seed': REFERENCE_SEED, 'parent': 'artifacts/EXP-0023/v2'}
    if c != expected:
        raise ValueError('EXP0025 configuration differs from registered v2')
    if np.__version__ != '2.3.3' or torch.__version__.split('+')[0] != '2.8.0':
        raise RuntimeError('Frozen NumPy/PyTorch versions required')


def _preflight(config: Path, c: dict[str, Any]) -> dict[str, Any]:
    if git_output('status', '--porcelain'):
        raise RuntimeError('Clean frozen checkout required')
    sources = source_fingerprints()
    suite_path = Path('artifacts/validation/suite.json')
    preflight_path = Path('artifacts/validation/exp0025_preflight.json')
    suite, preflight = _json(suite_path), _json(preflight_path)
    if suite.get('pass') is not True or suite.get('source_sha256') != sources:
        raise RuntimeError('Passing source-matched suite required')
    if (preflight.get('pass') is not True or preflight.get('source_sha256') != sources
            or preflight.get('config_sha256') != sha(config)):
        raise RuntimeError('Passing source/config-matched EXP0025 preflight required')
    parent = Path(c['parent'])
    manifest, completion = _json(parent / 'manifest.json'), _json(parent / 'completion.json')
    if ((manifest.get('experiment_id'), manifest.get('version')) != ('EXP-0023', 'v2') \
            or completion.get('status') != 'COMPLETED' or completion.get('research_updates') != 24000):
        raise RuntimeError('Frozen EXP0023 parent incomplete')
    old = {(r['seed'], r['variant']): r for r in completion['records']}
    if len(old) != 24 or any((s, a) not in old for s in range(23001, 23007)
                             for a in ('REL_BASE', 'REL_CYCLE', 'REL_EXTRA', 'INV_BASE')):
        raise RuntimeError('EXP0023 parent model matrix incomplete')
    parent_hashes = {str(parent / name): sha(parent / name)
                     for name in ('manifest.json', 'completion.json')}
    for arm in ARMS:
        parent_arm = arm + '_BASE'
        source = parent / str(REFERENCE_SEED) / parent_arm
        row, full = old[REFERENCE_SEED, parent_arm], _json(source / 'result.json')
        if {k: v for k, v in full.items() if k != 'losses'} != row:
            raise RuntimeError(f'Parent completion/result mismatch: {arm}')
        cp = source / 'checkpoint-1000.pt'
        if sha(cp) != full['curves'][-1]['checkpoint_sha256']:
            raise RuntimeError(f'Parent checkpoint changed: {arm}')
        parent_hashes[str(source / 'result.json')] = sha(source / 'result.json')
        parent_hashes[str(cp)] = sha(cp)
    frozen = (config, Path(c['protocol']), Path(c['entry_contract']), Path('uv.lock'),
              suite_path, preflight_path)
    return {'source_sha256': sources,
            'frozen_hashes': {str(path): sha(path) for path in frozen},
            'parent_artifact_sha256': parent_hashes,
            'parent_source_sha256': manifest['source_sha256']}


def _model(path: Path, seed: int, variant: str) -> LearnedMemory:
    state = torch.load(path, map_location='cpu', weights_only=True)
    arm = variant
    parent_variant = variant + '_BASE' if seed == REFERENCE_SEED else variant
    if any((state.get('seed') != seed, state.get('variant') != parent_variant,
            state.get('arm') != arm, state.get('step') != 1000)):
        raise RuntimeError(f'Wrong checkpoint metadata: {seed}/{variant}')
    model = LearnedMemory(arm, seed=seed)
    model.load_state_dict(state['model'])
    model.eval()
    return model


def _arrays(raw: dict[str, Any]) -> dict[str, np.ndarray]:
    if not isinstance(raw, dict):
        raise TypeError('Core must return a raw array dictionary')
    for key, value in raw.items():
        if not isinstance(value, np.ndarray) or not np.isfinite(value).all():
            raise FloatingPointError(f'Invalid raw array: {key}')
    return raw


def _resources(step: int, model: LearnedMemory, variant: str) -> dict[str, Any]:
    if not 0 <= step <= 1000:
        raise ValueError('Checkpoint step outside training range')
    nparams = sum(p.numel() for p in model.parameters())
    if nparams != PARAMETERS[variant]:
        raise RuntimeError('Frozen model parameter count changed')
    tokens = sum((min(step, stop) - previous) * 64 * length
                 for previous, stop, length in ((0, 125, 1), (125, 250, 2),
                                                (250, 500, 4), (500, 1000, 8))
                 if step > previous)
    return {'parameters': nparams,
            'parameter_bytes': sum(p.numel() * p.element_size() for p in model.parameters()),
            'state_values': 6, 'state_bytes_per_example': 24,
            'updates': step, 'main_examples': 64 * step,
            'training_transition_tokens': tokens,
            'executed_flops': 'NOT_MEASURED', 'peak_memory': 'NOT_MEASURED'}


def _train(seed: int, variant: str, monitor: dict[str, np.ndarray],
           path: Path, started: float) -> dict[str, Any]:
    path.mkdir(parents=True, exist_ok=False)
    arm = variant
    model = LearnedMemory(arm, seed=seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=.003, betas=(.9, .999),
                                 eps=1e-8, weight_decay=0)
    initial = {part: state_hash(model, part + '.') for part in ('encoder', 'readout')}
    initial['all'] = state_hash(model)
    history = hashlib.sha256()
    curves: list[dict[str, Any]] = []
    updates = 0
    arm_start, arm_cpu = time.monotonic(), time.process_time()

    def checkpoint(step: int) -> None:
        cp = path / f'checkpoint-{step}.pt'
        torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                    'seed': seed, 'arm': arm, 'variant': variant, 'step': step}, cp)
        model.eval()
        with torch.no_grad():
            z, g, _ = predict_full(model, monitor)
        if z.shape != (128, 9) or g.shape != (128, 9, 3) or not np.isfinite(z).all() or not np.isfinite(g).all():
            raise RuntimeError('Invalid monitor prediction')
        pred = path / f'monitor-{step}.npz'
        _npz(pred, {'logits': z, 'gram': g})
        from .refresh_replication_analysis import scores
        curves.append({'step': step, 'metrics': scores(z, monitor['targets'], monitor['commands']),
                       'checkpoint_sha256': sha(cp), 'prediction_sha256': sha(pred),
                       'wall_seconds': time.monotonic() - arm_start,
                       'cpu_seconds': time.process_time() - arm_cpu,
                       'resources': _resources(step, model, variant)})
        print(json.dumps({'seed': seed, 'arm': variant, 'monitor': step}), flush=True)

    try:
        checkpoint(0)
        for step in range(1, 1001):
            _budget(started)
            batch = generate(64, length_at(step), [252001, seed, step, 0], 'MIX')
            history.update(bytes.fromhex(data_hash(batch)))
            model.train()
            optimizer.zero_grad(set_to_none=True)
            z = model(*tensors(batch))
            loss = F.binary_cross_entropy_with_logits(z, torch.from_numpy(batch['targets']))
            if not torch.isfinite(loss).item():
                raise FloatingPointError('Nonfinite training loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            updates += 1
            if step in MONITOR_STEPS:
                checkpoint(step)
        row = {'seed': seed, 'arm': variant, 'origin': 'PROSPECTIVE_TRAINING',
               'updates': updates, 'initial_hashes': initial,
               'training_data_sha256': history.hexdigest(), 'curves': curves,
               'frozen_model_state_sha256': state_hash(model),
               'resources': _resources(updates, model, variant),
               'training_monitor_seconds': curves[-1]['wall_seconds'],
               'training_wall_seconds': time.monotonic() - arm_start,
               'training_cpu_seconds': time.process_time() - arm_cpu}
        if len(curves) != len(MONITOR_STEPS) or [c['step'] for c in curves] != list(MONITOR_STEPS):
            raise RuntimeError('Incomplete fixed monitor matrix')
        save_json(path / 'training-result.json', row)
        return row
    except BaseException as error:
        save_json(path / 'failure.json', {'error': repr(error), 'updates': updates,
                                         'curves': curves,
                                         'wall_seconds': time.monotonic() - arm_start,
                                         'cpu_seconds': time.process_time() - arm_cpu})
        torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                    'seed': seed, 'variant': variant, 'updates': updates}, path / 'failed.pt')
        raise


def _reference(arm: str, parent: Path, out: Path,
               provenance: dict[str, Any]) -> dict[str, Any]:
    seed = REFERENCE_SEED
    source = parent / str(seed) / (arm + '_BASE')
    target = out / str(seed) / arm
    target.mkdir(parents=True, exist_ok=False)
    for old, new in (('result.json', 'source-result.json'),
                     ('checkpoint-1000.pt', 'checkpoint-1000.pt')):
        _copy(source / old, target / new,
              provenance['parent_artifact_sha256'][str(source / old)])
    model = _model(target / 'checkpoint-1000.pt', seed, arm)
    return {'seed': seed, 'arm': arm, 'origin': 'FROZEN_REFERENCE', 'updates': 0,
            'parent_training_updates': 1000,
            'frozen_model_state_sha256': state_hash(model),
            'parent_source_result_sha256': sha(target / 'source-result.json'),
            'resources': {**_resources(0, model, arm), 'prototype_fits': 1}}


def _calibrate(row: dict[str, Any], data: dict[str, np.ndarray],
               out: Path) -> None:
    started, cpu_started = time.monotonic(), time.process_time()
    seed, arm = row['seed'], row['arm']
    path = out / str(seed) / arm
    model = _model(path / 'checkpoint-1000.pt', seed, arm)
    before = state_hash(model)
    with torch.no_grad():
        raw, summary = fit_prototypes(model, data)
    raw = _arrays(raw)
    if raw['prototypes'].shape != (2, 3, 2) or raw['prototype_grams'].shape != (2, 2, 2):
        raise RuntimeError('Wrong registered prototype shape')
    if raw['calibration_logits'].shape != (512, 9) or raw['calibration_gram'].shape != (512, 9, 3):
        raise RuntimeError('Wrong calibration forward shape')
    if raw['calibration_labels'].shape != (512,) or raw['eigenvalues'].shape != (2, 2):
        raise RuntimeError('Wrong calibration audit shape')
    if state_hash(model) != before or before != row['frozen_model_state_sha256']:
        raise RuntimeError('Frozen model changed during prototype fit')
    _npz(path / 'prototypes.npz', raw)
    row['calibration'] = summary
    row['calibration_data_sha256'] = sha(out / 'data' / str(seed) / 'calibration.npz')
    row['calibration_wall_seconds'] = time.monotonic() - started
    row['calibration_cpu_seconds'] = time.process_time() - cpu_started
    row.setdefault('artifact_sha256', {})['checkpoint-1000.pt'] = sha(path / 'checkpoint-1000.pt')
    row['artifact_sha256']['prototypes.npz'] = sha(path / 'prototypes.npz')
    row['resources'].update({'prototype_count': 2, 'prototype_values': 12,
                             'prototype_bytes': 48, 'prototype_fits': 1,
                             'calibration_transition_tokens': 512 * 8})


def _evaluate(model: LearnedMemory, data: dict[str, np.ndarray], condition: str,
              prototypes: np.ndarray, path: Path) -> dict[str, Any]:
    from .refresh_replication_analysis import scores, refresh_diagnostics
    started, cpu_started = time.monotonic(), time.process_time()
    with torch.no_grad():
        raw = _arrays(core.refresh_rollout(model, data, condition, prototypes))
    n, t = data['commands'].shape
    for key, shape in (('logits', (n, t + 1)), ('gram', (n, t + 1, 3)),
                       ('post_logits', (n, t + 1)), ('post_gram', (n, t + 1, 3)),
                       ('refresh_mask', (t + 1,)), ('selected_labels', (n, t + 1))):
        if raw[key].shape != shape:
            raise RuntimeError(f'Wrong {key} rollout shape')
    if condition == 'NONE':
        original, original_gram, _ = predict_full(model, data)
        if not np.array_equal(raw['logits'], original) or not np.array_equal(raw['gram'], original_gram):
            raise RuntimeError('NONE differs from original forward')
    _npz(path, raw)
    return {'metrics': scores(raw['logits'], data['targets'], data['commands']),
            'diagnostics': refresh_diagnostics(raw, condition),
            'artifact_sha256': sha(path),
            'wall_seconds': time.monotonic() - started,
            'cpu_seconds': time.process_time() - cpu_started,
            'transition_tokens': n * t,
            'refresh_calls': int(n * np.count_nonzero(raw['refresh_mask']))}


def run(config: Path, output_override: Path | None = None) -> Path:
    started, cpu_started = time.monotonic(), time.process_time()
    c = _json(config)
    validate_config(c)
    provenance = _preflight(config, c)
    if output_override is not None and not output_override.is_absolute():
        raise ValueError('EXP0025 --output override must be absolute')
    out = output_override or Path(c['output'])
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    save_json(out / 'manifest.json', {'experiment_id': 'EXP-0025', 'version': 'v2',
              'config': c, 'config_sha256': sha(config), 'git_commit': git_output('rev-parse', 'HEAD'),
              'git_status': '', 'actual_output': str(out.resolve()), 'python': sys.version,
              'torch': torch.__version__, 'numpy': np.__version__, 'platform': platform.platform(),
              'gate1_claim': False, 'matching_criterion_version': 'G1-MATCH-v3', **provenance})
    rows: list[dict[str, Any]] = []
    data_files: dict[str, str] = {}
    updates = 0
    qualification_count = evaluation_count = 0
    try:
        parent = Path(c['parent'])
        for name in ('manifest.json', 'completion.json'):
            _copy(parent / name, out / ('parent-' + name),
                  provenance['parent_artifact_sha256'][str(parent / name)])
        # Complete all training before supervised calibration or evaluation input generation.
        for seed in SEEDS:
            _budget(started)
            directory = out / 'data' / str(seed)
            directory.mkdir(parents=True, exist_ok=False)
            monitor = generate(128, 8, [251001, seed, 8, 0], 'MIX', .4, 0.)
            _npz(directory / 'monitor.npz', monitor)
            data_files[f'data/{seed}/monitor.npz'] = sha(directory / 'monitor.npz')
            pair = []
            for arm in ARMS:
                _budget(started)
                path = out / str(seed) / arm
                try:
                    row = _train(seed, arm, monitor, path, started)
                except BaseException:
                    failure = path / 'failure.json'
                    if failure.exists():
                        updates += _json(failure)['updates']
                    raise
                updates += row['updates']
                if updates > 24000:
                    raise RuntimeError('Registered research update cap exceeded')
                rows.append(row)
                pair.append(row)
            if pair[0]['training_data_sha256'] != pair[1]['training_data_sha256']:
                raise RuntimeError('Paired training data differ')
            for part in ('encoder', 'readout'):
                if pair[0]['initial_hashes'][part] != pair[1]['initial_hashes'][part]:
                    raise RuntimeError(f'Paired {part} initialization differs')
        if len(rows) != 24 or updates != 24000:
            raise RuntimeError('Incomplete prospective training matrix')
        for arm in ARMS:
            _budget(started)
            rows.append(_reference(arm, parent, out, provenance))
        # Entire codebook matrix is frozen before qualification/test inputs are generated.
        for seed in (*SEEDS, REFERENCE_SEED):
            _budget(started)
            directory = out / 'data' / str(seed)
            directory.mkdir(parents=True, exist_ok=(seed != REFERENCE_SEED))
            calibration = generate(512, 8, [253001, seed, 8, 0], 'MIX', .4, 0.)
            _npz(directory / 'calibration.npz', calibration)
            data_files[f'data/{seed}/calibration.npz'] = sha(directory / 'calibration.npz')
            for row in (r for r in rows if r['seed'] == seed):
                _budget(started)
                _calibrate(row, calibration, out)
        if len(rows) != 26 or sum(r['resources']['prototype_fits'] for r in rows) != 26:
            raise RuntimeError('Incomplete fixed prototype matrix')
        save_json(out / 'calibration_complete.json', {
            'records': [{'seed': r['seed'], 'arm': r['arm'], 'origin': r['origin'],
                         'calibration': r['calibration'],
                         'prototype_sha256': r['artifact_sha256']['prototypes.npz']} for r in rows],
            'data_files': dict(data_files), 'prototype_fits': 26, 'prototypes': 52,
            'research_updates': updates})
        for seed in (*SEEDS, REFERENCE_SEED):
            _budget(started)
            directory = out / 'data' / str(seed)
            qualification: dict[str, dict[str, np.ndarray]] = {}
            tests: dict[str, dict[str, np.ndarray]] = {}
            for cell, (length, mode, amplitude, mode_id) in CELLS.items():
                for phase, root, target in (('qual', 254001, qualification),
                                            ('test', 255001, tests)):
                    data = generate(512, length, [root, seed, length, mode_id], mode, amplitude, 0.)
                    name = f'{phase}_{cell}.npz'
                    _npz(directory / name, data)
                    data_files[f'data/{seed}/{name}'] = sha(directory / name)
                    target[cell] = data
            for row in (r for r in rows if r['seed'] == seed):
                path = out / str(seed) / row['arm']
                model = _model(path / 'checkpoint-1000.pt', seed, row['arm'])
                before = state_hash(model)
                with np.load(path / 'prototypes.npz') as loaded:
                    prototypes = loaded['prototypes'].copy()
                row['qualification'] = {'NONE': {}}
                for cell, data in qualification.items():
                    _budget(started)
                    row['qualification']['NONE'][cell] = _evaluate(
                        model, data, 'NONE', prototypes, path / f'qual_{cell}.npz')
                    qualification_count += 1
                row['evaluation'] = {}
                for condition in CONDITIONS:
                    row['evaluation'][condition] = {}
                    for cell, data in tests.items():
                        _budget(started)
                        row['evaluation'][condition][cell] = _evaluate(
                            model, data, condition, prototypes, path / 'test' / condition / f'{cell}.npz')
                        evaluation_count += 1
                if state_hash(model) != before or before != row['frozen_model_state_sha256']:
                    raise RuntimeError('Frozen neural weights changed during evaluation')
                save_json(path / 'result.json', row)
                print(json.dumps({'seed': seed, 'arm': row['arm'], 'status': 'COMPLETED'}), flush=True)
        tokens = sum(item['transition_tokens'] for r in rows
                     for phase in ('qualification', 'evaluation')
                     for condition in r[phase].values() for item in condition.values())
        training_tokens = sum(r['resources']['training_transition_tokens'] for r in rows)
        if (qualification_count != 104 or evaluation_count != 520
                or tokens != EVALUATION_TOKENS or training_tokens != TRAINING_TOKENS):
            raise RuntimeError('Registered evaluation/resource matrix changed')
        _budget(started)
        save_json(out / 'completion.json', {'status': 'COMPLETED', 'records': rows,
                  'data_files': data_files, 'trained_models': 24, 'reference_models': 2,
                  'model_count': 26, 'prototype_fits': 26,
                  'qualification_count': qualification_count,
                  'evaluation_count': evaluation_count,
                  'monitor_checkpoints': 120, 'research_updates': updates,
                  'training_transition_tokens': training_tokens,
                  'calibration_transition_tokens': 106496,
                  'evaluation_transition_tokens': tokens,
                  'wall_seconds': time.monotonic() - started,
                  'cpu_seconds': time.process_time() - cpu_started})
    except BaseException as error:
        save_json(out / 'run_failure.json', {'error': repr(error),
                  'research_updates': updates, 'trained_models_completed':
                  sum(r['origin'] == 'PROSPECTIVE_TRAINING' for r in rows),
                  'calibrated_models': sum('calibration' in r for r in rows),
                  'qualification_count': qualification_count,
                  'evaluation_count': evaluation_count,
                  'completed_rows': [[r['seed'], r['arm']] for r in rows if 'evaluation' in r],
                  'wall_seconds': time.monotonic() - started,
                  'cpu_seconds': time.process_time() - cpu_started})
        raise
    return out


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()

    def abort(*_: Any) -> None:
        raise TimeoutError('EXP0025 worker deadline')

    signal.signal(signal.SIGTERM, abort)
    run(args.config, args.output)
