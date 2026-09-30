"""EXP-0021 v1: frozen four-seed replication and checkpoint interventions."""
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

from .frozen_binary_probe import save_json
from .learned_protected_pilot import (
    ARMS, CELLS, KEYS, LearnedMemory, data_hash, generate, gram, length_at,
    metrics, predict, predict_with_gram, radial, state_hash, tensors,
)
from .order_observation import sha
from .protected_interventions import (
    intervention_data, intervention_metrics, intervention_rollout,
)
from .run import git_output, source_fingerprints


SEEDS = (20002, 20003, 20004, 20005)
PILOT_SEED = 20001
MONITOR_STEPS = (0, 25, 50, 75, 100, 125, 150, 200, 250, 375, 500, 750, 1000)
CONDITIONS = ('NONE', 'NORM', 'ANGLE', 'NORM_SHAM', 'ANGLE_SHAM')
PARAMETERS = {'RAW': 502, 'INV': 406, 'REL': 340}
WALL_CAP = 600


def validate_config(c: dict[str, Any]) -> None:
    """Fail closed on changes to registered choices, including pilot provenance."""
    frozen = {
        'experiment_id': 'EXP-0021', 'version': 'v1',
        'workflow': 'protected_replication',
        'protocol': 'experiments/EXP-0021/v1.md',
        'entry_contract': 'research/entries/EXP-0021.json',
        'output': 'artifacts/EXP-0021/v1', 'seeds': list(SEEDS),
        'pilot_seed': PILOT_SEED, 'pilot_source': 'artifacts/EXP-0020/v2',
        'arms': list(ARMS), 'monitor_steps': list(MONITOR_STEPS),
        'gate1_claim': False, 'matching_criterion_version': 'G1-MATCH-v3',
        'resource': {'worker_wall_seconds_cap': WALL_CAP,
                     'reproduction_wall_seconds_cap': 180,
                     'research_update_cap': 12000},
    }
    if c != frozen:
        raise ValueError('EXP-0021 configuration differs from frozen v1 registration')
    if np.__version__ != '2.3.3' or torch.__version__.split('+')[0] != '2.8.0':
        raise RuntimeError('Frozen NumPy/PyTorch versions required')


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _save_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as file:
        np.savez_compressed(file, **arrays)


def _budget(started: float) -> None:
    if time.monotonic() - started >= WALL_CAP:
        raise TimeoutError('EXP-0021 shared run and analysis wall budget exhausted')


def _preflight(config: Path) -> tuple[dict[str, Any], dict[str, str], dict[str, str]]:
    if git_output('status', '--porcelain'):
        raise RuntimeError('Clean frozen checkout required for EXP-0021')
    fingerprints = source_fingerprints()
    suite = Path('artifacts/validation/suite.json')
    receipt = Path('artifacts/validation/exp0021_preflight.json')
    validation = _load_json(suite)
    engineering = _load_json(receipt)
    if validation.get('pass') is not True or validation.get('source_sha256') != fingerprints:
        raise RuntimeError('Passing source-matched validation suite required')
    if engineering.get('pass') is not True:
        raise RuntimeError('Passing EXP-0021 engineering preflight required')
    source_digest = sha(Path('src/neurotopo/protected_interventions.py'))
    attested_source = engineering.get('model_source_sha256', engineering.get('source_sha256'))
    if attested_source != source_digest and attested_source != fingerprints:
        raise RuntimeError('Intervention source changed since preflight')
    if engineering.get('config_sha256') != sha(config):
        raise RuntimeError('EXP-0021 config changed since preflight')
    frozen_paths = [config, Path('experiments/EXP-0021/v1.md'),
                    Path('research/entries/EXP-0021.json'), Path('uv.lock'),
                    suite, receipt]
    frozen_hashes = {str(path): sha(path) for path in frozen_paths}
    pilot = Path('artifacts/EXP-0020/v2')
    source_manifest = _load_json(pilot / 'manifest.json')
    source_completion = _load_json(pilot / 'completion.json')
    if (source_manifest.get('experiment_id'), source_manifest.get('version')) != ('EXP-0020', 'v2'):
        raise RuntimeError('Wrong exploratory source experiment')
    if source_completion.get('status') != 'COMPLETED' or source_completion.get('updates') != 3000:
        raise RuntimeError('Exploratory pilot incomplete')
    if {r['arm'] for r in source_completion['records']} != set(ARMS):
        raise RuntimeError('Exploratory pilot arm set incomplete')
    source_paths = [pilot / name for name in ('manifest.json', 'completion.json', 'analysis.json')]
    for arm in ARMS:
        row = _load_json(pilot / arm / 'result.json')
        completion_row = next(r for r in source_completion['records'] if r['arm'] == arm)
        cp = pilot / arm / 'checkpoint-1000.pt'
        if row['arm'] != arm or row['updates'] != 1000 or len(row['losses']) != 1000:
            raise RuntimeError(f'Invalid exploratory pilot result for {arm}')
        if row['curves'][-1]['step'] != 1000 or row['curves'][-1]['checkpoint_sha256'] != sha(cp):
            raise RuntimeError(f'Exploratory pilot checkpoint hash mismatch for {arm}')
        if {k: v for k, v in row.items() if k != 'losses'} != completion_row:
            raise RuntimeError(f'Exploratory pilot result differs from completion for {arm}')
        source_paths.extend((pilot / arm / 'result.json', cp))
    pilot_hashes = {str(path): sha(path) for path in source_paths}
    return source_completion, frozen_hashes, pilot_hashes


def _datasets(seed: int, data_dir: Path) -> tuple[dict[str, np.ndarray], dict[str, dict[str, np.ndarray]], dict[str, np.ndarray]]:
    data_dir.mkdir(parents=True, exist_ok=False)
    monitor = generate(128, 8, [204001, seed, 8, 0], 'MIX')
    _save_npz(data_dir / 'monitor.npz', monitor)
    evaluation = {}
    for cell, (length, mode, amplitude, leak) in CELLS.items():
        dataset = generate(1024, length, [203001, seed, length, 1 if mode == 'HOLD' else 2],
                           mode, amplitude, leak)
        _save_npz(data_dir / f'{cell}.npz', dataset)
        evaluation[cell] = dataset
    intervention = intervention_data(seed)
    if set(intervention) != set(KEYS) or len(intervention['b0']) != 1024:
        raise RuntimeError('Invalid registered intervention dataset')
    _save_npz(data_dir / 'intervention.npz', intervention)
    hashes = {path.name: sha(path) for path in data_dir.glob('*.npz')
              if not path.name.startswith('._')}
    if len(hashes) != 10:
        raise RuntimeError('Missing shared input dataset')
    return monitor, evaluation, intervention


def _resource_at(step: int, model: LearnedMemory) -> dict[str, int | str]:
    lengths = ((125, 1), (250, 2), (500, 4), (1000, 8))
    if not 0 <= step <= 1000:
        raise ValueError('Checkpoint step outside registered training range')
    previous, tokens = 0, 0
    for stop, length in lengths:
        count = max(0, min(step, stop) - previous)
        tokens += count * 64 * length
        previous = stop
    parameters = sum(p.numel() for p in model.parameters())
    if parameters != PARAMETERS[model.arm]:
        raise RuntimeError('Frozen parameter count changed')
    return {'parameters': parameters, 'parameter_bytes': sum(p.numel() * p.element_size() for p in model.parameters()),
            'state_values': 6, 'state_bytes_per_example': 6 * 4,
            'examples': 64 * step, 'operation_tokens': tokens, 'updates': step,
            'executed_flops': 'NOT_MEASURED', 'peak_memory': 'NOT_MEASURED'}


def _evaluate(model: LearnedMemory, datasets: dict[str, dict[str, np.ndarray]],
              intervention: dict[str, np.ndarray], path: Path, started: float
              ) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]], dict[str, str]]:
    evaluation: dict[str, dict[str, Any]] = {}
    predictions: dict[str, np.ndarray] = {}
    for cell, data in datasets.items():
        _budget(started)
        logits, grams = predict_with_gram(model, data)
        predictions[cell] = logits
        predictions[cell + '_gram'] = grams
        score = metrics(logits, data)
        truth = data['targets'][:, -1]
        means = np.stack([grams[truth == bit, -1].astype(float).mean(0) for bit in (0, 1)])
        weight = model.readout.weight.detach().numpy()[0].astype(float)
        score['final_gram_mean_by_b'] = means.tolist()
        score['mean_logit_separation_terms'] = (weight * (means[1] - means[0])).tolist()
        evaluation[cell] = score
    _save_npz(path / 'predictions.npz', predictions)
    arrays = intervention_rollout(model, intervention, seed=int(path.parent.name))
    if set(arrays) != set(CONDITIONS):
        raise RuntimeError('Intervention rollout condition set changed')
    intervention_scores: dict[str, dict[str, Any]] = {}
    artifact_hashes = {'predictions.npz': sha(path / 'predictions.npz')}
    for condition in CONDITIONS:
        _budget(started)
        output = arrays[condition]
        required = {'logits', 'gram', 'before', 'after', 'displacement',
                    'norm_sham_residual', 'norm_sham_unreachable', 'rotation'}
        if not required <= set(output):
            raise RuntimeError(f'Missing {condition} intervention raw arrays')
        if output['logits'].shape != (1024, 33) or output['gram'].shape != (1024, 33, 3):
            raise RuntimeError(f'Wrong {condition} intervention output shape')
        if not all(np.isfinite(value).all() for value in output.values()):
            raise FloatingPointError(f'Nonfinite {condition} intervention output')
        _save_npz(path / f'intervention-{condition}.npz', output)
        intervention_scores[condition] = intervention_metrics(output, intervention)
        artifact_hashes[f'intervention-{condition}.npz'] = sha(path / f'intervention-{condition}.npz')
    return evaluation, intervention_scores, artifact_hashes


def _train(seed: int, arm: str, monitor: dict[str, np.ndarray],
           datasets: dict[str, dict[str, np.ndarray]], intervention: dict[str, np.ndarray],
           path: Path, started: float) -> dict[str, Any]:
    path.mkdir(parents=True, exist_ok=False)
    model = LearnedMemory(arm, seed=seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=.003, betas=(.9, .999), eps=1e-8, weight_decay=0)
    initial = {'encoder': state_hash(model, 'encoder.'),
               'readout': state_hash(model, 'readout.'), 'all': state_hash(model)}
    losses: list[float] = []
    curves: list[dict[str, Any]] = []
    history = hashlib.sha256()
    arm_started = time.monotonic()
    cpu_started = time.process_time()

    def checkpoint(step: int) -> None:
        checkpoint_path = path / f'checkpoint-{step}.pt'
        torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                    'seed': seed, 'arm': arm, 'step': step}, checkpoint_path)
        logits = predict(model, monitor)
        monitor_path = path / f'monitor-{step}.npz'
        _save_npz(monitor_path, {'logits': logits})
        curve = {'step': step, 'metrics': metrics(logits, monitor),
                 'wall_seconds': time.monotonic() - arm_started,
                 'cpu_seconds': time.process_time() - cpu_started,
                 'resources': _resource_at(step, model),
                 'checkpoint_sha256': sha(checkpoint_path),
                 'prediction_sha256': sha(monitor_path)}
        curves.append(curve)
        print(json.dumps({'seed': seed, 'arm': arm, 'monitor': curve}), flush=True)

    try:
        checkpoint(0)
        for step in range(1, 1001):
            _budget(started)
            batch = generate(64, length_at(step), [202001, seed, step, 0], 'MIX')
            history.update(bytes.fromhex(data_hash(batch)))
            model.train()
            optimizer.zero_grad(set_to_none=True)
            logits = model(*tensors(batch))
            loss = F.binary_cross_entropy_with_logits(logits, torch.from_numpy(batch['targets']))
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite training loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            losses.append(float(loss.detach()))
            if step in MONITOR_STEPS:
                checkpoint(step)
        evaluation, interventions, hashes = _evaluate(model, datasets, intervention, path, started)
        record = {'seed': seed, 'arm': arm, 'origin': 'CONFIRMATION', 'updates': 1000,
                  'initial_hashes': initial, 'training_data_sha256': history.hexdigest(),
                  'losses': losses, 'curves': curves, 'evaluation': evaluation,
                  'intervention': interventions, 'artifact_sha256': hashes,
                  'learned_readout_weight': model.readout.weight.detach().numpy()[0].tolist(),
                  'learned_readout_bias': float(model.readout.bias.detach()[0]),
                  'learned_initial_gram': gram(radial(model.encoder.weight.reshape(2, 3, 2))).detach().numpy().tolist(),
                  'training_monitor_seconds': curves[-1]['wall_seconds'],
                  'arm_seconds': time.monotonic() - arm_started,
                  'arm_cpu_seconds': time.process_time() - cpu_started,
                  'resources': _resource_at(1000, model)}
        save_json(path / 'result.json', record)
        return record
    except BaseException as error:
        torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                    'seed': seed, 'arm': arm, 'updates': len(losses)}, path / 'failed.pt')
        save_json(path / 'failure.json', {'error': repr(error), 'updates': len(losses),
                                         'losses': losses, 'curves': curves})
        raise


def _import_pilot(seed: int, arm: str, datasets: dict[str, dict[str, np.ndarray]],
                  intervention: dict[str, np.ndarray], path: Path,
                  started: float) -> dict[str, Any]:
    arm_started = time.monotonic()
    cpu_started = time.process_time()
    path.mkdir(parents=True, exist_ok=False)
    source = Path('artifacts/EXP-0020/v2') / arm
    source_record = _load_json(source / 'result.json')
    shutil.copyfile(source / 'checkpoint-1000.pt', path / 'checkpoint-1000.pt')
    shutil.copyfile(source / 'result.json', path / 'source-result.json')
    if sha(path / 'checkpoint-1000.pt') != source_record['curves'][-1]['checkpoint_sha256']:
        raise RuntimeError('Copied exploratory checkpoint changed')
    checkpoint = torch.load(path / 'checkpoint-1000.pt', map_location='cpu', weights_only=True)
    if checkpoint['arm'] != arm or checkpoint['step'] != 1000:
        raise RuntimeError('Wrong pilot checkpoint metadata')
    model = LearnedMemory(arm, seed=seed)
    model.load_state_dict(checkpoint['model'])
    evaluation, interventions, hashes = _evaluate(model, datasets, intervention, path, started)
    for cell in CELLS:
        actual, previous = evaluation[cell], source_record['evaluation'][cell]
        if set(actual) != set(previous):
            raise RuntimeError(f'Pilot score schema changed for {arm}/{cell}')
        for key, value in actual.items():
            old = previous[key]
            if value is None:
                equal = old is None
            elif isinstance(value, list):
                equal = bool(np.allclose(value, old, rtol=0, atol=1e-8))
            elif isinstance(value, (float, int)):
                equal = abs(value - old) <= 1e-8
            else:
                equal = value == old
            if not equal:
                raise RuntimeError(f'Pilot replay mismatch for {arm}/{cell}/{key}')
    hashes.update({'checkpoint-1000.pt': sha(path / 'checkpoint-1000.pt'),
                   'source-result.json': sha(path / 'source-result.json')})
    record = {'seed': seed, 'arm': arm, 'origin': 'EXPLORATORY_PILOT', 'updates': 0,
              'source_training_updates': source_record['updates'],
              'source_training_data_sha256': source_record['training_data_sha256'],
              'source_initial_hashes': source_record['initial_hashes'],
              'curves': [], 'evaluation': evaluation, 'intervention': interventions,
              'artifact_sha256': hashes, 'source_evaluation': source_record['evaluation'],
              'learned_readout_weight': source_record['learned_readout_weight'],
              'learned_readout_bias': source_record['learned_readout_bias'],
              'arm_seconds': time.monotonic() - arm_started,
              'arm_cpu_seconds': time.process_time() - cpu_started,
              'resources': {**_resource_at(0, model), 'updates': 0,
                            'source_training_updates': 1000,
                            'source_training_examples': 64000,
                            'source_training_operation_tokens': 64 * (125 + 250 + 1000 + 4000)}}
    save_json(path / 'result.json', record)
    return record


def run(config: Path, output_override: Path | None = None) -> Path:
    started = time.monotonic()
    cpu_started = time.process_time()
    c = _load_json(config)
    validate_config(c)
    source_completion, frozen_hashes, pilot_hashes = _preflight(config)
    out = output_override or Path(c['output'])
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    manifest = {'experiment_id': 'EXP-0021', 'version': 'v1', 'config': c,
                'git_commit': git_output('rev-parse', 'HEAD'), 'git_status': '',
                'source_sha256': source_fingerprints(), 'frozen_hashes': frozen_hashes,
                'pilot_artifact_sha256': pilot_hashes, 'pilot_source_commit':
                _load_json(Path(c['pilot_source']) / 'manifest.json')['git_commit'],
                'python': sys.version, 'torch': torch.__version__, 'numpy': np.__version__,
                'platform': platform.platform(), 'gate1_claim': False,
                'matching_criterion_version': 'G1-MATCH-v3'}
    save_json(out / 'manifest.json', manifest)
    rows: list[dict[str, Any]] = []
    completed_updates = 0
    try:
        for seed in (*SEEDS, PILOT_SEED):
            _budget(started)
            seed_dir = out / str(seed)
            seed_dir.mkdir(parents=True, exist_ok=False)
            monitor, datasets, intervention = _datasets(seed, seed_dir / 'data')
            for arm in ARMS:
                path = seed_dir / arm
                if seed == PILOT_SEED:
                    row = _import_pilot(seed, arm, datasets, intervention, path, started)
                else:
                    try:
                        row = _train(seed, arm, monitor, datasets, intervention, path, started)
                    except BaseException:
                        failure = path / 'failure.json'
                        if failure.exists():
                            completed_updates += _load_json(failure)['updates']
                        raise
                    completed_updates += row['updates']
                rows.append(row)
                print(json.dumps({'seed': seed, 'arm': arm,
                                  'evaluation': row['evaluation'],
                                  'intervention': row['intervention']}), flush=True)
            trio = rows[-3:]
            if len({r.get('training_data_sha256', r.get('source_training_data_sha256')) for r in trio}) != 1:
                raise RuntimeError(f'Unequal training streams within seed {seed}')
            if len({r.get('initial_hashes', r.get('source_initial_hashes'))['encoder'] for r in trio}) != 1:
                raise RuntimeError(f'Unequal encoder initialization within seed {seed}')
            if len({r.get('initial_hashes', r.get('source_initial_hashes'))['readout'] for r in trio}) != 1:
                raise RuntimeError(f'Unequal readout initialization within seed {seed}')
        if completed_updates != 12000:
            raise RuntimeError('Wrong completed research update count')
        data_hashes = {str(path.relative_to(out)): sha(path)
                       for path in out.glob('*/data/*.npz')
                       if not path.name.startswith('._')}
        save_json(out / 'completion.json', {'status': 'COMPLETED',
                  'seconds': time.monotonic() - started,
                  'cpu_seconds': time.process_time() - cpu_started,
                  'updates': completed_updates,
                  'records': [{k: v for k, v in row.items() if k != 'losses'} for row in rows],
                  'data_files': data_hashes,
                  'pilot_source_records': {r['arm']: r['training_data_sha256']
                                           for r in source_completion['records']}})
    except BaseException as error:
        save_json(out / 'run_failure.json', {'error': repr(error),
                  'seconds': time.monotonic() - started,
                  'cpu_seconds': time.process_time() - cpu_started,
                  'updates': completed_updates,
                  'completed_rows': [(r['seed'], r['arm']) for r in rows]})
        raise
    return out


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()

    def abort(*_: Any) -> None:
        raise TimeoutError('EXP-0021 worker deadline')

    signal.signal(signal.SIGTERM, abort)
    run(args.config, args.output)
