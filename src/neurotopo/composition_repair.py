"""Frozen EXP-0023 v2 runner. Decisions belong to composition_repair_analysis."""
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

from . import composition_core as core
from .frozen_binary_probe import save_json
from .learned_protected_pilot import (
    KEYS, LearnedMemory, data_hash, generate, length_at, metrics, state_hash,
)
from .order_observation import sha
from .run import git_output, source_fingerprints


OLD_SEEDS = (20001, 20002, 20003, 20004, 20005)
SEEDS = (23001, 23002, 23003, 23004, 23005, 23006)
OLD_ARMS = ('RAW', 'INV', 'REL')
ARMS = ('REL_BASE', 'REL_CYCLE', 'REL_EXTRA', 'INV_BASE')
MONITOR_STEPS = (0, 125, 250, 500, 1000)
CELLS = {
    'IID_H': (8, 'HOLD', .4), 'IID_U': (8, 'UPDATE', .4),
    'LONG64_H': (64, 'HOLD', np.pi), 'LONG64_U': (64, 'UPDATE', np.pi),
    'LONG128_H': (128, 'HOLD', np.pi), 'LONG128_U': (128, 'UPDATE', np.pi),
    'LONG256_H': (256, 'HOLD', np.pi), 'LONG256_U': (256, 'UPDATE', np.pi),
}
CAP = 1200
MAX_UPDATES = 24000
PARAMETERS = {'REL': 340, 'INV': 406}


def validate_config(c: dict[str, Any]) -> None:
    expected = {
        'experiment_id': 'EXP-0023', 'version': 'v2',
        'workflow': 'composition_repair', 'protocol': 'experiments/EXP-0023/v2.md',
        'entry_contract': 'research/entries/EXP-0023.json',
        'output': 'artifacts/EXP-0023/v2', 'parent': 'artifacts/EXP-0021/v1',
        'old_seeds': list(OLD_SEEDS), 'seeds': list(SEEDS), 'arms': list(ARMS),
        'coefficient': .1, 'updates': 1000, 'batch_size': 64,
        'monitor_steps': list(MONITOR_STEPS), 'evaluation_examples': 512,
        'diagnostic_examples': 128, 'gate1_claim': False,
        'matching_criterion_version': 'G1-MATCH-v3',
        'resource': {'worker_wall_seconds_cap': CAP,
                     'reproduction_wall_seconds_cap': 300,
                     'research_update_cap': MAX_UPDATES,
                     'general_validation_overfit_cap': 900,
                     'special_fit_update_cap': 90},
    }
    if c != expected:
        raise ValueError('EXP-0023 configuration differs from frozen v2 registration')
    if np.__version__ != '2.3.3' or torch.__version__.split('+')[0] != '2.8.0':
        raise RuntimeError('Frozen NumPy/PyTorch versions required')


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        np.savez_compressed(stream, **arrays)


def _budget(started: float) -> None:
    if time.monotonic() - started >= CAP:
        raise TimeoutError('EXP-0023 shared run and analysis wall budget exhausted')


def _checkpoint_source(parent: Path, seed: int, arm: str, row: dict[str, Any]) -> tuple[Path, Path, str]:
    source = parent / str(seed) / arm
    result = source / 'result.json'
    full = _json(result)
    if {k: v for k, v in full.items() if k != 'losses'} != row:
        raise RuntimeError(f'Parent result/completion disagreement: {seed}/{arm}')
    cp = source / 'checkpoint-1000.pt'
    expected = (full['artifact_sha256']['checkpoint-1000.pt'] if seed == 20001
                else full['curves'][-1]['checkpoint_sha256'])
    if sha(cp) != expected:
        raise RuntimeError(f'Parent checkpoint hash mismatch: {seed}/{arm}')
    return result, cp, expected


def _preflight(config: Path, c: dict[str, Any]) -> dict[str, Any]:
    if git_output('status', '--porcelain'):
        raise RuntimeError('Clean frozen checkout required for EXP-0023')
    sources = source_fingerprints()
    suite_path = Path('artifacts/validation/suite.json')
    preflight_path = Path('artifacts/validation/exp0023_preflight.json')
    suite, preflight = _json(suite_path), _json(preflight_path)
    if suite.get('pass') is not True or suite.get('source_sha256') != sources:
        raise RuntimeError('Passing source-matched validation suite required')
    if (preflight.get('pass') is not True or preflight.get('source_sha256') != sources
            or preflight.get('config_sha256') != sha(config)):
        raise RuntimeError('Passing EXP-0023 source/config-matched preflight required')
    frozen_paths = (config, Path(c['protocol']), Path(c['entry_contract']),
                    Path('uv.lock'), suite_path, preflight_path)
    parent = Path(c['parent'])
    parent_manifest = _json(parent / 'manifest.json')
    parent_completion = _json(parent / 'completion.json')
    if ((parent_manifest.get('experiment_id'), parent_manifest.get('version')) != ('EXP-0021', 'v1')
            or parent_completion.get('status') != 'COMPLETED'
            or parent_completion.get('updates') != 12000):
        raise RuntimeError('EXP-0021 parent is not complete')
    rows = parent_completion['records']
    expected_rows = {(seed, arm) for seed in OLD_SEEDS for arm in OLD_ARMS}
    if len(rows) != 15 or {(r['seed'], r['arm']) for r in rows} != expected_rows:
        raise RuntimeError('EXP-0021 parent model set is incomplete')
    parent_hashes = {str(parent / name): sha(parent / name)
                     for name in ('manifest.json', 'completion.json')}
    for row in rows:
        result, cp, digest = _checkpoint_source(parent, row['seed'], row['arm'], row)
        parent_hashes[str(result)] = sha(result)
        parent_hashes[str(cp)] = digest
    return {'source_sha256': sources,
            'frozen_hashes': {str(path): sha(path) for path in frozen_paths},
            'parent_artifact_sha256': parent_hashes,
            'parent_source_sha256': parent_manifest['source_sha256']}


def _copy_checked(source: Path, destination: Path, expected: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open('xb') as stream, source.open('rb') as original:
        shutil.copyfileobj(original, stream)
    if sha(destination) != expected:
        raise RuntimeError(f'Copied frozen artifact changed: {source}')


def _data(seed: int, data_dir: Path) -> tuple[dict[str, np.ndarray],
                                              dict[str, dict[str, np.ndarray]],
                                              dict[str, np.ndarray], dict[str, str]]:
    data_dir.mkdir(parents=True, exist_ok=False)
    monitor = generate(128, 8, [236001, seed, 8, 0], 'MIX')
    diagnostic = generate(128, 8, [234001, seed, 8, 0], 'MIX')
    _npz(data_dir / 'monitor.npz', monitor)
    _npz(data_dir / 'diagnostic.npz', diagnostic)
    evaluation: dict[str, dict[str, np.ndarray]] = {}
    for name, (length, mode, amp) in CELLS.items():
        data = generate(512, length, [233001, seed, length, 1 if mode == 'HOLD' else 2],
                        mode, amp, 0.)
        _npz(data_dir / f'{name}.npz', data)
        evaluation[name] = data
    hashes = {path.name: sha(path) for path in data_dir.glob('*.npz')
              if not path.name.startswith('._')}
    if len(hashes) != 10:
        raise RuntimeError('Missing fixed input dataset')
    return monitor, evaluation, diagnostic, hashes


def _diagnostic(model: LearnedMemory, data: dict[str, np.ndarray], path: Path
                ) -> tuple[dict[str, Any], str]:
    model.eval()
    with torch.no_grad():
        raw, summary = core.diagnostic(model, data)
    if not isinstance(raw, dict) or not isinstance(summary, dict):
        raise RuntimeError('Diagnostic core returned invalid records')
    for key, value in raw.items():
        if not isinstance(value, np.ndarray) or not np.isfinite(value).all():
            raise FloatingPointError(f'Invalid diagnostic raw array {key}')
    _npz(path / 'diagnostic.npz', raw)
    return summary, sha(path / 'diagnostic.npz')


def _old_diagnostics(out: Path, parent: Path, parent_rows: list[dict[str, Any]],
                     started: float) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    lookup = {(r['seed'], r['arm']): r for r in parent_rows}
    for seed in OLD_SEEDS:
        _budget(started)
        data_dir = out / 'old' / str(seed) / 'data'
        data_dir.mkdir(parents=True, exist_ok=False)
        data = generate(128, 8, [234001, seed, 8, 0], 'MIX')
        _npz(data_dir / 'diagnostic.npz', data)
        for arm in OLD_ARMS:
            _budget(started)
            path = out / 'old' / str(seed) / arm
            path.mkdir(parents=True, exist_ok=False)
            result_source, cp_source, cp_hash = _checkpoint_source(parent, seed, arm, lookup[seed, arm])
            _copy_checked(result_source, path / 'source-result.json', sha(result_source))
            _copy_checked(cp_source, path / 'checkpoint-1000.pt', cp_hash)
            state = torch.load(path / 'checkpoint-1000.pt', map_location='cpu', weights_only=True)
            if state['arm'] != arm or state['step'] != 1000 or ('seed' in state and state['seed'] != seed):
                raise RuntimeError(f'Wrong parent checkpoint metadata: {seed}/{arm}')
            model = LearnedMemory(arm, seed=seed)
            model.load_state_dict(state['model'])
            before_hash = state_hash(model)
            summary, raw_hash = _diagnostic(model, data, path)
            if state_hash(model) != before_hash:
                raise RuntimeError('Frozen parent model changed during diagnosis')
            row = {'seed': seed, 'arm': arm, 'origin': 'FROZEN_PARENT_DESCRIPTIVE',
                   'research_updates': 0, 'diagnostic': summary,
                   'frozen_model_state_sha256': before_hash,
                   'parent_evaluation': lookup[seed, arm]['evaluation'],
                   'artifact_sha256': {'checkpoint-1000.pt': cp_hash,
                                       'source-result.json': sha(path / 'source-result.json'),
                                       'diagnostic.npz': raw_hash},
                   'diagnostic_data_sha256': sha(data_dir / 'diagnostic.npz')}
            save_json(path / 'result.json', row)
            rows.append(row)
            print(json.dumps({'old_seed': seed, 'arm': arm, 'status': 'DIAGNOSED'}), flush=True)
    return rows


def _resources(step: int, model: LearnedMemory, variant: str) -> dict[str, Any]:
    main_tokens = sum((min(step, stop) - previous) * 64 * length
                      for previous, stop, length in ((0, 125, 1), (125, 250, 2),
                                                     (250, 500, 4), (500, 1000, 8))
                      if step > previous)
    if not 0 <= step <= 1000:
        raise ValueError('Checkpoint step outside registered training range')
    extra = variant in ('REL_CYCLE', 'REL_EXTRA')
    nparams = sum(p.numel() for p in model.parameters())
    if nparams != PARAMETERS[model.arm]:
        raise RuntimeError('Frozen model parameter count changed')
    return {'parameters': nparams,
            'parameter_bytes': sum(p.numel() * p.element_size() for p in model.parameters()),
            'state_values': 6, 'state_bytes_per_example': 24,
            'updates': step, 'main_examples': 64 * step,
            'main_operation_tokens': main_tokens,
            'extra_transition_tokens': 64 * 3 * step if extra else 0,
            'extra_branch_readout_heads': 64 * 2 * step if extra else 0,
            'executed_flops': 'NOT_MEASURED',
            'operator_profile': 'SEE_PREFLIGHT_RECEIPT',
            'peak_memory': 'NOT_MEASURED'}


def _score(logits: np.ndarray, data: dict[str, np.ndarray]) -> dict[str, Any]:
    score = metrics(logits, data)
    z = logits[:, 1:].astype(np.float64)
    target = data['targets'][:, 1:].astype(np.float64)
    score['future_bce'] = float(np.mean(np.logaddexp(0., z) - target * z))
    return score


def _evaluate(model: LearnedMemory, datasets: dict[str, dict[str, np.ndarray]],
              path: Path, started: float) -> tuple[dict[str, dict[str, Any]], str]:
    predictions: dict[str, np.ndarray] = {}
    scores: dict[str, dict[str, Any]] = {}
    model.eval()
    with torch.no_grad():
        for cell, data in datasets.items():
            _budget(started)
            z, g, _ = core.predict_full(model, data)
            if z.shape != (512, data['commands'].shape[1] + 1) or g.shape != (512, z.shape[1], 3):
                raise RuntimeError(f'Invalid prediction shape for {cell}')
            if not np.isfinite(z).all() or not np.isfinite(g).all():
                raise FloatingPointError(f'Nonfinite prediction for {cell}')
            predictions[cell] = z
            predictions[cell + '_gram'] = g
            scores[cell] = _score(z, data)
    _npz(path / 'predictions.npz', predictions)
    return scores, sha(path / 'predictions.npz')


def _train(seed: int, variant: str, monitor: dict[str, np.ndarray],
           evaluation: dict[str, dict[str, np.ndarray]], diagnostic: dict[str, np.ndarray],
           path: Path, started: float) -> dict[str, Any]:
    path.mkdir(parents=True, exist_ok=False)
    arm = 'INV' if variant == 'INV_BASE' else 'REL'
    model = LearnedMemory(arm, seed=seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=.003, betas=(.9, .999),
                                 eps=1e-8, weight_decay=0)
    initial = {'encoder': state_hash(model, 'encoder.'),
               'readout': state_hash(model, 'readout.'), 'all': state_hash(model)}
    losses: list[dict[str, float]] = []
    curves: list[dict[str, Any]] = []
    history = hashlib.sha256()
    arm_start, arm_cpu = time.monotonic(), time.process_time()
    updates = 0

    def checkpoint(step: int) -> None:
        cp = path / f'checkpoint-{step}.pt'
        torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                    'seed': seed, 'arm': arm, 'variant': variant, 'step': step}, cp)
        model.eval()
        with torch.no_grad():
            z, g, _ = core.predict_full(model, monitor)
        if z.shape != (128, 9) or g.shape != (128, 9, 3):
            raise RuntimeError('Invalid monitor prediction shape')
        monitor_path = path / f'monitor-{step}.npz'
        _npz(monitor_path, {'logits': z, 'gram': g})
        curve = {'step': step, 'metrics': _score(z, monitor),
                 'wall_seconds': time.monotonic() - arm_start,
                 'cpu_seconds': time.process_time() - arm_cpu,
                 'resources': _resources(step, model, variant),
                 'checkpoint_sha256': sha(cp), 'prediction_sha256': sha(monitor_path)}
        curves.append(curve)
        print(json.dumps({'seed': seed, 'variant': variant, 'monitor': step}), flush=True)

    try:
        checkpoint(0)
        for step in range(1, 1001):
            _budget(started)
            batch = generate(64, length_at(step), [232001, seed, step, 0], 'MIX')
            history.update(bytes.fromhex(data_hash(batch)))
            model.train()
            optimizer.zero_grad(set_to_none=True)
            total, main, aux = core.loss_terms(model, batch, variant, coefficient=.1)
            if not all(torch.isfinite(x).item() for x in (total, main, aux)):
                raise FloatingPointError('Nonfinite training loss')
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            updates += 1
            losses.append({'total': float(total.detach()), 'main': float(main.detach()),
                           'aux': float(aux.detach())})
            if step in MONITOR_STEPS:
                checkpoint(step)
        scores, predictions_hash = _evaluate(model, evaluation, path, started)
        summary, diagnostic_hash = _diagnostic(model, diagnostic, path)
        row = {'seed': seed, 'variant': variant, 'arm': arm, 'origin': 'PROSPECTIVE_TRAINING',
               'updates': updates, 'initial_hashes': initial,
               'training_data_sha256': history.hexdigest(), 'losses': losses,
               'curves': curves, 'evaluation': scores, 'diagnostic': summary,
               'artifact_sha256': {'predictions.npz': predictions_hash,
                                   'diagnostic.npz': diagnostic_hash},
               'resources': _resources(updates, model, variant),
               'training_monitor_seconds': curves[-1]['wall_seconds'],
               'arm_seconds': time.monotonic() - arm_start,
               'arm_cpu_seconds': time.process_time() - arm_cpu}
        save_json(path / 'result.json', row)
        return row
    except BaseException as error:
        save_json(path / 'failure.json', {'error': repr(error), 'updates': updates,
                                         'losses': losses, 'curves': curves,
                                         'wall_seconds': time.monotonic() - arm_start,
                                         'cpu_seconds': time.process_time() - arm_cpu})
        torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                    'seed': seed, 'variant': variant, 'updates': updates}, path / 'failed.pt')
        raise


def run(config: Path, output_override: Path | None = None) -> Path:
    start, cpu_start = time.monotonic(), time.process_time()
    c = _json(config)
    validate_config(c)
    provenance = _preflight(config, c)
    if output_override is not None and not output_override.is_absolute():
        raise ValueError('EXP-0023 --output override must be absolute')
    out = output_override or Path(c['output'])
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    manifest = {'experiment_id': 'EXP-0023', 'version': 'v2', 'config': c,
                'config_sha256': sha(config), 'git_commit': git_output('rev-parse', 'HEAD'),
                'git_status': '', 'actual_output': str(out.resolve()), 'python': sys.version, 'torch': torch.__version__,
                'numpy': np.__version__, 'platform': platform.platform(),
                'gate1_claim': False, 'matching_criterion_version': 'G1-MATCH-v3',
                **provenance}
    save_json(out / 'manifest.json', manifest)
    parent = Path(c['parent'])
    old_rows: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    data_files: dict[str, str] = {}
    completed_updates = 0
    try:
        _copy_checked(parent / 'manifest.json', out / 'parent-manifest.json',
                      provenance['parent_artifact_sha256'][str(parent / 'manifest.json')])
        _copy_checked(parent / 'completion.json', out / 'parent-completion.json',
                      provenance['parent_artifact_sha256'][str(parent / 'completion.json')])
        old_rows = _old_diagnostics(out, parent, _json(parent / 'completion.json')['records'], start)
        for seed in SEEDS:
            _budget(start)
            monitor, evaluation, diagnostic, hashes = _data(seed, out / 'data' / str(seed))
            data_files.update({f'data/{seed}/{name}': digest for name, digest in hashes.items()})
            for variant in ARMS:
                _budget(start)
                path = out / str(seed) / variant
                try:
                    row = _train(seed, variant, monitor, evaluation, diagnostic, path, start)
                except BaseException:
                    if (path / 'failure.json').exists():
                        completed_updates += _json(path / 'failure.json')['updates']
                    raise
                completed_updates += row['updates']
                if completed_updates > MAX_UPDATES:
                    raise RuntimeError('Registered research update cap exceeded')
                rows.append(row)
                print(json.dumps({'seed': seed, 'variant': variant, 'status': 'COMPLETED'}), flush=True)
            quartet = rows[-4:]
            if len({r['training_data_sha256'] for r in quartet}) != 1:
                raise RuntimeError(f'Paired training data mismatch for seed {seed}')
            if len({r['initial_hashes']['all'] for r in quartet[:3]}) != 1:
                raise RuntimeError(f'Paired REL initialization mismatch for seed {seed}')
            for part in ('encoder', 'readout'):
                if len({r['initial_hashes'][part] for r in quartet}) != 1:
                    raise RuntimeError(f'Four-arm {part} initialization mismatch for seed {seed}')
        if len(old_rows) != 15 or len(rows) != 24 or completed_updates != MAX_UPDATES:
            raise RuntimeError('Incomplete EXP-0023 registered model set')
        _budget(start)
        save_json(out / 'completion.json', {'status': 'COMPLETED', 'old_records': old_rows,
                  'records': [{k: v for k, v in row.items() if k != 'losses'} for row in rows],
                  'data_files': data_files, 'old_models': 15, 'trained_models': 24,
                  'old_research_updates': 0, 'research_updates': completed_updates,
                  'wall_seconds': time.monotonic() - start,
                  'cpu_seconds': time.process_time() - cpu_start})
    except BaseException as error:
        save_json(out / 'run_failure.json', {'error': repr(error),
                  'research_updates': completed_updates,
                  'old_models_completed': len(list((out / 'old').glob('*/*/result.json'))),
                  'trained_models_completed': len(rows),
                  'completed_rows': [[r['seed'], r['variant']] for r in rows],
                  'wall_seconds': time.monotonic() - start,
                  'cpu_seconds': time.process_time() - cpu_start})
        raise
    return out


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()

    def abort(*_: Any) -> None:
        raise TimeoutError('EXP-0023 worker deadline')

    signal.signal(signal.SIGTERM, abort)
    run(args.config, args.output)
