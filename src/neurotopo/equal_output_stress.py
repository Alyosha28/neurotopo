"""EXP-0022 v2 frozen checkpoint runner. No training or optimizer is used."""
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

from . import equal_output_geometry as geometry
from .frozen_binary_probe import save_json
from .learned_protected_pilot import ARMS, LearnedMemory, generate, state_hash
from .order_observation import sha
from .run import git_output, source_fingerprints

SEEDS = (20002, 20003, 20004, 20005, 20001)
RADII = (0.01, 0.05, 0.15)
CONDITIONS = ('NONE', 'RANDOM', 'SELECTED', 'SHAM')
CAP = 900


def validate_config(c: dict[str, Any]) -> None:
    expected = dict(experiment_id='EXP-0022', version='v2', workflow='equal_output_stress',
                    protocol='experiments/EXP-0022/v2.md',
                    entry_contract='research/entries/EXP-0022.json',
                    output='artifacts/EXP-0022/v2', parent='artifacts/EXP-0021/v1',
                    seeds=list(SEEDS), arms=list(ARMS), radii=list(RADII),
                    prefixes=64, candidates=16, search_suffixes=4, test_suffixes=16,
                    suffix_length=32, gate1_claim=False,
                    matching_criterion_version='G1-MATCH-v3',
                    resource=dict(worker_wall_seconds_cap=900,
                                  reproduction_wall_seconds_cap=300,
                                  research_update_cap=0,
                                  general_validation_overfit_cap=900))
    if c != expected:
        raise ValueError('EXP-0022 configuration differs from frozen v2 registration')
    if np.__version__ != '2.3.3' or torch.__version__.split('+')[0] != '2.8.0':
        raise RuntimeError('Frozen NumPy/PyTorch versions required')


def _npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        np.savez_compressed(stream, **arrays)


def _budget(start: float) -> None:
    if time.monotonic() - start >= CAP:
        raise TimeoutError('EXP-0022 shared run and analysis wall budget exhausted')


def _preflight(config: Path, c: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    if git_output('status', '--porcelain'):
        raise RuntimeError('Clean frozen checkout required for EXP-0022')
    source = source_fingerprints()
    suite_path = Path('artifacts/validation/suite.json')
    pre_path = Path('artifacts/validation/exp0022_preflight.json')
    suite = json.loads(suite_path.read_text())
    pre = json.loads(pre_path.read_text())
    if suite.get('pass') is not True or suite.get('source_sha256') != source:
        raise RuntimeError('Passing source-matched validation suite required')
    if pre.get('pass') is not True or pre.get('source_sha256') != source or pre.get('config_sha256') != sha(config):
        raise RuntimeError('Passing EXP-0022 source/config-matched preflight required')
    frozen = {str(path): sha(path) for path in
              (config, Path(c['protocol']), Path(c['entry_contract']), Path('uv.lock'),
               suite_path, pre_path)}
    parent = Path(c['parent'])
    manifest_path, completion_path = parent / 'manifest.json', parent / 'completion.json'
    parent_manifest = json.loads(manifest_path.read_text())
    completed = json.loads(completion_path.read_text())
    if ((parent_manifest.get('experiment_id'), parent_manifest.get('version')) != ('EXP-0021', 'v1')
            or completed.get('status') != 'COMPLETED' or completed.get('updates') != 12000):
        raise RuntimeError('EXP-0021 parent is not complete')
    rows = completed['records']
    if len(rows) != 15 or {(r['seed'], r['arm']) for r in rows} != {(s, a) for s in SEEDS for a in ARMS}:
        raise RuntimeError('EXP-0021 parent checkpoint set is incomplete')
    parent_hashes = {str(path): sha(path) for path in (manifest_path, completion_path)}
    for row in rows:
        seed, arm = row['seed'], row['arm']
        result_path = parent / str(seed) / arm / 'result.json'
        full = json.loads(result_path.read_text())
        if {key: value for key, value in full.items() if key != 'losses'} != row:
            raise RuntimeError(f'Parent result/completion disagreement: {seed}/{arm}')
        cp = parent / str(seed) / arm / 'checkpoint-1000.pt'
        expected = (full['curves'][-1]['checkpoint_sha256'] if seed != 20001
                    else full['artifact_sha256']['checkpoint-1000.pt'])
        if sha(cp) != expected:
            raise RuntimeError(f'Parent checkpoint hash mismatch: {seed}/{arm}')
        parent_hashes[str(result_path)] = sha(result_path)
        parent_hashes[str(cp)] = expected
    return {'source_sha256': source, 'frozen_hashes': frozen,
            'parent_artifact_sha256': parent_hashes,
            'parent_source_sha256': parent_manifest['source_sha256']}, completed


def _load_model(parent: Path, seed: int, arm: str) -> tuple[LearnedMemory, Path]:
    cp = parent / str(seed) / arm / 'checkpoint-1000.pt'
    state = torch.load(cp, map_location='cpu', weights_only=True)
    if state['arm'] != arm or state['step'] != 1000 or ('seed' in state and state['seed'] != seed):
        raise RuntimeError(f'Wrong checkpoint metadata: {seed}/{arm}')
    model = LearnedMemory(arm, seed=seed)
    model.load_state_dict(state['model'])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, cp


def _bank_arrays(bank: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    required = {'states', 'sham', 'valid', 'sham_valid', 'logit_error',
                'radius_error', 'norm_error', 'directions', 'eigbasis'}
    missing = required - set(bank)
    if missing:
        raise RuntimeError(f'Geometry candidate bank lacks {sorted(missing)}')
    return {key: np.asarray(value) for key, value in bank.items()}


def run(config: Path, output_override: Path | None = None) -> Path:
    start, cpu = time.monotonic(), time.process_time()
    c = json.loads(config.read_text())
    validate_config(c)
    provenance, parent_completion = _preflight(config, c)
    out = output_override or Path(c['output'])
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    manifest = dict(experiment_id='EXP-0022', version='v2', config=c,
                    config_sha256=sha(config), git_commit=git_output('rev-parse', 'HEAD'),
                    git_status='', python=sys.version, torch=torch.__version__,
                    numpy=np.__version__, platform=platform.platform(),
                    gate1_claim=False, matching_criterion_version='G1-MATCH-v3', **provenance)
    save_json(out / 'manifest.json', manifest)
    records: list[dict[str, Any]] = []
    data_files: dict[str, str] = {}
    try:
        with torch.no_grad():
            for seed in SEEDS:
                _budget(start)
                seed_start, seed_cpu = time.monotonic(), time.process_time()
                data_dir = out / str(seed) / 'data'
                prefix = generate(64, 32, [222001, seed, 32, 2], 'UPDATE', np.pi, 0.)
                _npz(data_dir / 'prefix.npz', **prefix)
                bits = prefix['targets'][:, -1].astype(np.int64)
                search = geometry.suffix_data(seed, bits, 4, 222002)
                _npz(data_dir / 'search.npz', **search)
                prepared: list[tuple[dict[str, Any], LearnedMemory, np.ndarray,
                                     dict[str, np.ndarray], np.ndarray, np.ndarray, str]] = []
                for arm in ARMS:
                    _budget(start)
                    model, cp = _load_model(Path(c['parent']), seed, arm)
                    arm_dir = out / str(seed) / arm
                    arm_dir.mkdir(parents=True, exist_ok=False)
                    shutil.copyfile(cp, arm_dir / 'checkpoint.pt')
                    if sha(arm_dir / 'checkpoint.pt') != sha(cp):
                        raise RuntimeError('Copied checkpoint hash mismatch')
                    before_hash = state_hash(model)
                    h = geometry.prefix_states(model, prefix)
                    if np.asarray(h).shape != (64, 3, 2):
                        raise RuntimeError('Wrong prefix state shape')
                    for radius in RADII:
                        _budget(start)
                        cell_start, cell_cpu = time.monotonic(), time.process_time()
                        bank = _bank_arrays(geometry.candidate_bank(model, h, seed, radius))
                        if bank['states'].shape != (64, 16, 3, 2) or bank['sham'].shape != (64, 3, 2):
                            raise RuntimeError('Wrong geometry bank shape')
                        search_states = np.where(bank['valid'][:, :, None, None],
                                                 bank['states'], np.asarray(h)[:, None])
                        search_logits, _ = geometry.rollout(model, search_states, search)
                        indices, scores = geometry.select_candidates(search_logits, search['targets'], bank['valid'])
                        indices = np.asarray(indices, dtype=np.int64)
                        scores = np.asarray(scores)
                        if indices.shape != (64,) or scores.shape != (64, 16) or np.any((indices < 0) | (indices >= 16)):
                            raise RuntimeError('Wrong search selection shape or range')
                        cell_dir = arm_dir / f'r{round(radius * 1000):03d}'
                        cell_dir.mkdir(parents=True, exist_ok=False)
                        # Durable search selection receipt precedes test-data generation.
                        _npz(cell_dir / 'selection.npz', indices=indices, scores=scores,
                             valid=bank['valid'], sham_valid=bank['sham_valid'])
                        prepared.append((dict(seed=seed, arm=arm, radius=radius,
                                              path=str(cell_dir.relative_to(out)),
                                              origin='EXPLORATORY_PILOT' if seed == 20001 else 'PRIOR_CONFIRMATION_SEED',
                                              search_wall_seconds=time.monotonic() - cell_start,
                                              search_cpu_seconds=time.process_time() - cell_cpu),
                                         model, np.asarray(h), bank, indices, scores, before_hash))
                # No test target or command exists in the selection phase above.
                test = geometry.suffix_data(seed, bits, 16, 222003)
                _npz(data_dir / 'test.npz', **test)
                for name in ('prefix.npz', 'search.npz', 'test.npz'):
                    path = data_dir / name
                    data_files[str(path.relative_to(out))] = sha(path)
                for base, model, h, bank, indices, scores, before_hash in prepared:
                    _budget(start)
                    cell_start, cell_cpu = time.monotonic(), time.process_time()
                    n = np.arange(64)
                    random_valid = np.asarray(bank['valid'][:, 0], dtype=bool)
                    selected_valid = np.asarray(bank['valid'][n, indices], dtype=bool)
                    sham_valid = np.asarray(bank['sham_valid'], dtype=bool)
                    states = np.stack((h, np.where(random_valid[:, None, None], bank['states'][:, 0], h),
                                       np.where(selected_valid[:, None, None], bank['states'][n, indices], h),
                                       np.where(sham_valid[:, None, None], bank['sham'], h)), axis=1)
                    test_logits, test_gram = geometry.rollout(model, states, test)
                    cell_dir = out / base['path']
                    raw = dict(prefix_states=h, condition_states=states, candidate_states=bank['states'],
                               sham_state=bank['sham'], selection_indices=indices,
                               selection_scores=scores,
                               valid=bank['valid'], sham_valid=bank['sham_valid'],
                               condition_valid=np.stack((np.ones(64, bool), random_valid,
                                                         selected_valid, sham_valid), axis=1),
                               test_logits=test_logits, test_gram=test_gram)
                    raw.update({key: value for key, value in bank.items()
                                if key not in raw and key not in ('states', 'sham')})
                    _npz(cell_dir / 'raw.npz', **raw)
                    after_hash = state_hash(model)
                    if after_hash != before_hash:
                        raise RuntimeError('Frozen parent model state changed during evaluation')
                    record = {**base, 'artifact_sha256': {'selection.npz': sha(cell_dir / 'selection.npz'),
                                                          'raw.npz': sha(cell_dir / 'raw.npz')},
                              'parent_checkpoint_sha256': sha(cell_dir.parent / 'checkpoint.pt'),
                              'model_state_hash_before': before_hash,
                              'model_state_hash_after': after_hash,
                              'test_write_wall_seconds': time.monotonic() - cell_start,
                              'test_write_cpu_seconds': time.process_time() - cell_cpu,
                              'resources': {'parameters': sum(p.numel() for p in model.parameters()),
                                            'parameter_bytes': sum(p.numel() * p.element_size() for p in model.parameters()),
                                            'state_values': 6, 'state_bytes_per_example': 24,
                                            'candidate_states': 64 * 16,
                                            'search_suffix_rollouts': 64 * 16 * 4,
                                            'test_suffix_rollouts': 64 * 4 * 16,
                                            'research_updates': 0, 'optimizer_updates': 0,
                                            'executed_flops': 'NOT_MEASURED',
                                            'peak_memory': 'NOT_MEASURED'}}
                    save_json(cell_dir / 'record.json', record)
                    record['artifact_sha256']['record.json'] = sha(cell_dir / 'record.json')
                    records.append(record)
                    print(json.dumps({'seed': base['seed'], 'arm': base['arm'],
                                      'radius': base['radius'], 'status': 'RECORDED'}), flush=True)
                save_json(out / str(seed) / 'seed_timing.json',
                          {'wall_seconds': time.monotonic() - seed_start,
                           'cpu_seconds': time.process_time() - seed_cpu})
        if len(records) != 45:
            raise RuntimeError('Expected 45 model-radius cells')
        _budget(start)
        save_json(out / 'completion.json', {'status': 'COMPLETED', 'records': records,
                  'data_files': data_files, 'research_updates': 0, 'optimizer_updates': 0,
                  'candidate_states': 45 * 64 * 16,
                  'search_suffix_rollouts': 45 * 64 * 16 * 4,
                  'test_suffix_rollouts': 45 * 64 * 4 * 16,
                  'wall_seconds': time.monotonic() - start,
                  'cpu_seconds': time.process_time() - cpu,
                  'parent_records': len(parent_completion['records'])})
    except BaseException as error:
        save_json(out / 'run_failure.json', {'error': repr(error),
                  'wall_seconds': time.monotonic() - start,
                  'cpu_seconds': time.process_time() - cpu,
                  'completed_cells': [(r['seed'], r['arm'], r['radius']) for r in records],
                  'research_updates': 0})
        raise
    return out


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()

    def abort(*_: Any) -> None:
        raise TimeoutError('EXP-0022 worker deadline')

    signal.signal(signal.SIGTERM, abort)
    run(args.config, args.output)
