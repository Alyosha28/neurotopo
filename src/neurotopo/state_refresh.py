"""Frozen EXP0024v1 runner; scientific decisions live in state_refresh_analysis."""
from __future__ import annotations

import argparse
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

from . import state_refresh_core as core
from .composition_repair import SEEDS
from .frozen_binary_probe import save_json
from .learned_protected_pilot import LearnedMemory, generate, state_hash
from .order_observation import sha
from .run import git_output, source_fingerprints

ARMS = ('REL_BASE', 'INV_BASE')
CONDITIONS = ('NONE', 'GAUGE8', 'RESET8', 'RESET1')
CELLS = {'IID_H': (8, 'HOLD', .4, 1), 'IID_U': (8, 'UPDATE', .4, 2),
         'LONG_H': (256, 'HOLD', np.pi, 1), 'LONG_U': (256, 'UPDATE', np.pi, 2)}
CAP = 900


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        np.savez_compressed(stream, **arrays)


def _copy(source: Path, dest: Path, digest: str) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open('xb') as target, source.open('rb') as original:
        shutil.copyfileobj(original, target)
    if sha(dest) != digest:
        raise RuntimeError(f'Frozen copy hash mismatch: {source}')


def _budget(started: float) -> None:
    if time.monotonic() - started >= CAP:
        raise TimeoutError('EXP0024 run and analysis 900-second budget exhausted')


def validate_config(c: dict[str, Any]) -> None:
    expected = {'experiment_id': 'EXP-0024', 'version': 'v1', 'workflow': 'state_refresh',
                'protocol': 'experiments/EXP-0024/v1.md',
                'entry_contract': 'research/entries/EXP-0024.json',
                'output': 'artifacts/EXP-0024/v1', 'parent': 'artifacts/EXP-0023/v2',
                'seeds': list(SEEDS), 'arms': list(ARMS), 'conditions': list(CONDITIONS),
                'examples': 512, 'primary_seed': 23005, 'gate1_claim': False,
                'matching_criterion_version': 'G1-MATCH-v3',
                'resource': {'worker_wall_seconds_cap': 900,
                             'reproduction_wall_seconds_cap': 300,
                             'research_update_cap': 0,
                             'general_validation_overfit_cap': 900}}
    if c != expected:
        raise ValueError('EXP0024 config differs from registered v1')
    if np.__version__ != '2.3.3' or torch.__version__.split('+')[0] != '2.8.0':
        raise RuntimeError('Frozen NumPy/PyTorch versions required')


def _preflight(config: Path, c: dict[str, Any]) -> dict[str, Any]:
    if git_output('status', '--porcelain'):
        raise RuntimeError('Clean frozen checkout required')
    sources = source_fingerprints()
    suite_path = Path('artifacts/validation/suite.json')
    preflight_path = Path('artifacts/validation/exp0024_preflight.json')
    suite, preflight = _json(suite_path), _json(preflight_path)
    if suite.get('pass') is not True or suite.get('source_sha256') != sources:
        raise RuntimeError('Passing source-matched suite required')
    if (preflight.get('pass') is not True or preflight.get('source_sha256') != sources
            or preflight.get('config_sha256') != sha(config)):
        raise RuntimeError('Passing source/config-matched EXP0024 preflight required')
    parent = Path(c['parent'])
    manifest, completion = _json(parent / 'manifest.json'), _json(parent / 'completion.json')
    if ((manifest.get('experiment_id'), manifest.get('version')) != ('EXP-0023', 'v2') \
            or completion.get('status') != 'COMPLETED' or completion.get('research_updates') != 24000):
        raise RuntimeError('Frozen EXP0023 parent incomplete')
    rows = {(r['seed'], r['variant']): r for r in completion['records']}
    if len(rows) != 24 or any((s, a) not in rows for s in SEEDS for a in ARMS):
        raise RuntimeError('Parent model matrix incomplete')
    parent_hashes = {str(parent / n): sha(parent / n) for n in ('manifest.json', 'completion.json')}
    for seed in SEEDS:
        for arm in ARMS:
            source = parent / str(seed) / arm
            row, full = rows[seed, arm], _json(source / 'result.json')
            if {k: v for k, v in full.items() if k != 'losses'} != row:
                raise RuntimeError(f'Parent completion/result mismatch: {seed}/{arm}')
            cp = source / 'checkpoint-1000.pt'
            if sha(cp) != full['curves'][-1]['checkpoint_sha256']:
                raise RuntimeError(f'Parent checkpoint changed: {seed}/{arm}')
            parent_hashes[str(source / 'result.json')] = sha(source / 'result.json')
            parent_hashes[str(cp)] = sha(cp)
    frozen = (config, Path(c['protocol']), Path(c['entry_contract']), Path('uv.lock'),
              suite_path, preflight_path)
    return {'source_sha256': sources, 'frozen_hashes': {str(p): sha(p) for p in frozen},
            'parent_artifact_sha256': parent_hashes,
            'parent_source_sha256': manifest['source_sha256']}


def _model(path: Path, seed: int, variant: str) -> LearnedMemory:
    state = torch.load(path, map_location='cpu', weights_only=True)
    arm = 'REL' if variant == 'REL_BASE' else 'INV'
    if any((state.get('seed') != seed, state.get('variant') != variant,
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
            raise FloatingPointError(f'Invalid core raw array: {key}')
    return raw


def _calibrate(seed: int, variant: str, parent: Path, out: Path,
               data: dict[str, np.ndarray], provenance: dict[str, Any]) -> dict[str, Any]:
    started, cpu_started = time.monotonic(), time.process_time()
    path = out / str(seed) / variant
    path.mkdir(parents=True, exist_ok=False)
    source = parent / str(seed) / variant
    for name in ('result.json', 'checkpoint-1000.pt'):
        _copy(source / name, path / ('source-result.json' if name == 'result.json' else name),
              provenance['parent_artifact_sha256'][str(source / name)])
    model = _model(path / 'checkpoint-1000.pt', seed, variant)
    before = state_hash(model)
    with torch.no_grad():
        raw, summary = core.fit_prototypes(model, data)
    raw = _arrays(raw)
    if raw['prototypes'].shape != (2, 3, 2) or raw['prototype_grams'].shape != (2, 2, 2):
        raise RuntimeError('Wrong registered prototype shape')
    if raw['calibration_logits'].shape != (512, 9) or raw['calibration_gram'].shape != (512, 9, 3):
        raise RuntimeError('Wrong calibration forward shape')
    if raw['calibration_labels'].shape != (512,) or raw['eigenvalues'].shape != (2, 2):
        raise RuntimeError('Wrong calibration audit shape')
    if state_hash(model) != before:
        raise RuntimeError('Frozen model changed during prototype fit')
    _npz(path / 'prototypes.npz', raw)
    with torch.no_grad():
        transitions = _arrays(core.transition_table(model, raw['prototypes']))
    _npz(path / 'transition_table.npz', transitions)
    return {'seed': seed, 'arm': variant, 'frozen_model_state_sha256': before,
            'calibration': summary, 'calibration_data_sha256': sha(out / 'data' / str(seed) / 'calibration.npz'),
            'calibration_wall_seconds': time.monotonic() - started,
            'calibration_cpu_seconds': time.process_time() - cpu_started,
            'artifact_sha256': {'source-result.json': sha(path / 'source-result.json'),
                                'checkpoint-1000.pt': sha(path / 'checkpoint-1000.pt'),
                                'prototypes.npz': sha(path / 'prototypes.npz'),
                                'transition_table.npz': sha(path / 'transition_table.npz')},
            'resources': {'parameters': sum(p.numel() for p in model.parameters()),
                          'parameter_bytes': sum(p.numel() * p.element_size() for p in model.parameters()),
                          'state_values': 6, 'state_bytes_per_example': 24,
                          'prototype_count': 2, 'prototype_values': 12, 'prototype_bytes': 48,
                          'prototype_fits': 1, 'optimizer_updates': 0,
                          'calibration_transition_tokens': 512 * 8,
                          'static_transition_tokens': 4,
                          'executed_flops': 'NOT_MEASURED', 'peak_memory': 'NOT_MEASURED'}}


def _evaluate(model: LearnedMemory, data: dict[str, np.ndarray], condition: str,
              prototypes: np.ndarray, path: Path) -> dict[str, Any]:
    from .state_refresh_analysis import scores, refresh_diagnostics
    start, cpu = time.monotonic(), time.process_time()
    with torch.no_grad():
        raw = _arrays(core.refresh_rollout(model, data, condition, prototypes))
    n, t = data['commands'].shape
    if raw['logits'].shape != (n, t + 1) or raw['gram'].shape != (n, t + 1, 3):
        raise RuntimeError('Wrong pre-refresh rollout shape')
    if raw['post_logits'].shape != (n, t + 1) or raw['post_gram'].shape != (n, t + 1, 3):
        raise RuntimeError('Wrong post-refresh rollout shape')
    if raw['refresh_mask'].shape != (t + 1,) or raw['selected_labels'].shape != (n, t + 1):
        raise RuntimeError('Wrong intervention audit shape')
    _npz(path, raw)
    return {'metrics': scores(raw['logits'], data['targets'], data['commands']),
            'diagnostics': refresh_diagnostics(raw, condition),
            'artifact_sha256': sha(path), 'wall_seconds': time.monotonic() - start,
            'cpu_seconds': time.process_time() - cpu,
            'transition_tokens': n * t,
            'refresh_calls': int(n * np.count_nonzero(raw['refresh_mask']))}


def run(config: Path, output_override: Path | None = None) -> Path:
    started, cpu_started = time.monotonic(), time.process_time()
    c = _json(config)
    validate_config(c)
    provenance = _preflight(config, c)
    if output_override is not None and not output_override.is_absolute():
        raise ValueError('EXP0024 --output override must be absolute')
    out = output_override or Path(c['output'])
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    save_json(out / 'manifest.json', {'experiment_id': 'EXP-0024', 'version': 'v1',
              'config': c, 'config_sha256': sha(config), 'git_commit': git_output('rev-parse', 'HEAD'),
              'git_status': '', 'actual_output': str(out.resolve()), 'python': sys.version,
              'torch': torch.__version__, 'numpy': np.__version__, 'platform': platform.platform(),
              'gate1_claim': False, 'matching_criterion_version': 'G1-MATCH-v3', **provenance})
    rows: list[dict[str, Any]] = []
    data_files: dict[str, str] = {}
    try:
        parent = Path(c['parent'])
        for name in ('manifest.json', 'completion.json'):
            _copy(parent / name, out / ('parent-' + name),
                  provenance['parent_artifact_sha256'][str(parent / name)])
        # This entire pass completes and hashes supervised calibration before any evaluation input exists.
        for seed in SEEDS:
            _budget(started)
            directory = out / 'data' / str(seed)
            directory.mkdir(parents=True, exist_ok=False)
            calibration = generate(512, 8, [242001, seed, 8, 0], 'MIX', .4, 0.)
            _npz(directory / 'calibration.npz', calibration)
            data_files[f'data/{seed}/calibration.npz'] = sha(directory / 'calibration.npz')
            for variant in ARMS:
                _budget(started)
                rows.append(_calibrate(seed, variant, parent, out, calibration, provenance))
        if len(rows) != 12 or sum(r['resources']['prototype_fits'] for r in rows) != 12:
            raise RuntimeError('Incomplete fixed prototype matrix')
        save_json(out / 'calibration_complete.json', {'records': [
            {'seed': r['seed'], 'arm': r['arm'], 'calibration': r['calibration'],
             'prototype_sha256': r['artifact_sha256']['prototypes.npz']} for r in rows],
            'data_files': dict(data_files), 'prototype_fits': 12, 'prototypes': 24,
            'optimizer_updates': 0})
        eval_count = 0
        for seed in SEEDS:
            _budget(started)
            directory = out / 'data' / str(seed)
            datasets = {}
            for cell, (length, mode, amplitude, mode_id) in CELLS.items():
                data = generate(512, length, [243001, seed, length, mode_id], mode, amplitude, 0.)
                _npz(directory / f'{cell}.npz', data)
                data_files[f'data/{seed}/{cell}.npz'] = sha(directory / f'{cell}.npz')
                datasets[cell] = data
            for row in (r for r in rows if r['seed'] == seed):
                path = out / str(seed) / row['arm']
                model = _model(path / 'checkpoint-1000.pt', seed, row['arm'])
                before = state_hash(model)
                with np.load(path / 'prototypes.npz') as loaded:
                    prototypes = loaded['prototypes'].copy()
                row['evaluation'] = {}
                for condition in CONDITIONS:
                    row['evaluation'][condition] = {}
                    for cell, data in datasets.items():
                        _budget(started)
                        record = _evaluate(model, data, condition, prototypes,
                                           path / condition / f'{cell}.npz')
                        row['evaluation'][condition][cell] = record
                        eval_count += 1
                if state_hash(model) != before or before != row['frozen_model_state_sha256']:
                    raise RuntimeError('Frozen neural weights changed during evaluation')
                save_json(path / 'result.json', row)
                print(json.dumps({'seed': seed, 'arm': row['arm'], 'status': 'COMPLETED'}), flush=True)
        if eval_count != 192 or sum(r['evaluation'][c][cell]['transition_tokens']
                                    for r in rows for c in CONDITIONS for cell in CELLS) != 12976128:
            raise RuntimeError('Registered evaluation count changed')
        _budget(started)
        save_json(out / 'completion.json', {'status': 'COMPLETED', 'records': rows,
                  'data_files': data_files, 'model_count': 12, 'prototype_fits': 12,
                  'evaluation_count': eval_count, 'research_updates': 0,
                  'evaluation_transition_tokens': 12976128,
                  'calibration_transition_tokens': 49152,
                  'wall_seconds': time.monotonic() - started,
                  'cpu_seconds': time.process_time() - cpu_started})
    except BaseException as error:
        save_json(out / 'run_failure.json', {'error': repr(error),
                  'completed_models': [[r['seed'], r['arm']] for r in rows if 'evaluation' in r],
                  'calibrated_models': [[r['seed'], r['arm']] for r in rows],
                  'research_updates': 0, 'wall_seconds': time.monotonic() - started,
                  'cpu_seconds': time.process_time() - cpu_started})
        raise
    return out


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()

    def abort(*_: Any) -> None:
        raise TimeoutError('EXP0024 worker deadline')

    signal.signal(signal.SIGTERM, abort)
    run(args.config, args.output)
