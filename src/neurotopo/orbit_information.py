"""EXP-0015 immutable zero-training task-information screen."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np

from .run import git_output, source_fingerprints, write_json


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def array_sha(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def validate_config(c: dict[str, Any]) -> None:
    expected = {
        'experiment_id': 'EXP-0015', 'version': 'v1', 'workflow': 'orbit_information',
        'gate1_claim': False,
        'task': dict(states=16, length=64, examples=16384, root=77000007,
                     stay=.7, forward=.2, backward=.1, emission_correct=.6),
        'information': dict(uniform_kl_mean_min=.2, suffix1_kl_ci_lower_min=.05,
                            suffix8_kl_ci_lower_min=.02, bootstrap_resamples=10000,
                            bootstrap_root=77100007, ci_quantiles=[.025, .975], bootstrap_block=32),
        'resource': dict(worker_wall_seconds_cap=300, reproduction_wall_seconds_cap=300,
                         research_update_cap=0),
    }
    for key, value in expected.items():
        if c.get(key) != value:
            raise ValueError(f'Frozen EXP-0015 contract differs: {key}')
    if np.__version__ != '2.3.3':
        raise ValueError('EXP-0015 requires pinned NumPy 2.3.3')


def generate(examples: int, length: int, root: int, states: int = 16) -> dict[str, np.ndarray]:
    initial, transition, emission = [np.random.Generator(np.random.PCG64(s))
                                   for s in np.random.SeedSequence(root).spawn(3)]
    z0 = initial.integers(0, states, size=examples).astype(np.int16)
    u, v = transition.random((examples, length)), emission.random((examples, length))
    z = np.empty((examples, length), dtype=np.int16)
    state = z0.copy()
    for t in range(length):
        increment = np.where(u[:, t] < .7, 0, np.where(u[:, t] < .9, 1, -1))
        state = ((state + increment) % states).astype(np.int16)
        z[:, t] = state
    wrong = v >= .6
    j = np.zeros(v.shape, dtype=np.int16)
    j[wrong] = np.floor((v[wrong]-.6)/.4*(states-1)).astype(np.int16)
    if (j[wrong] >= states-1).any():
        raise ValueError('Emission categorical index overflow')
    y = np.where(wrong, (z+1+j) % states, z).astype(np.int16)
    return dict(z0=z0, z=z, y=y)


def posterior(observations: np.ndarray, states: int = 16) -> np.ndarray:
    """Shift-based forward algorithm; time columns never access later observations."""
    if observations.ndim != 2 or not np.issubdtype(observations.dtype, np.integer):
        raise ValueError('Expected integer observation matrix')
    if ((observations < 0) | (observations >= states)).any():
        raise ValueError('Observation outside state alphabet')
    q = np.full((len(observations), states), 1/states, dtype=np.float64)
    labels = np.arange(states)[None, :]
    for column in observations.T:
        predicted = .7*q + .2*np.roll(q, 1, axis=1) + .1*np.roll(q, -1, axis=1)
        weights = np.where(labels == column[:, None], .6, .4/(states-1))
        q = predicted*weights
        q /= q.sum(axis=1, keepdims=True)
    if (not np.isfinite(q).all() or (q <= 0).any()
            or np.max(np.abs(q.sum(axis=1)-1)) > 1e-12):
        raise FloatingPointError('Invalid oracle posterior')
    return q


def kl(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    result = np.sum(p*(np.log(p)-np.log(q)), axis=1)
    if not np.isfinite(result).all() or result.min() < -1e-12:
        raise FloatingPointError('Invalid oracle-relative KL')
    return result


def bootstrap_means(values: np.ndarray, root: int, repetitions: int, block: int = 32) -> np.ndarray:
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError('Finite two-dimensional bootstrap data required')
    rng = np.random.Generator(np.random.PCG64(root))
    result = np.empty((repetitions, values.shape[1]), dtype=np.float64)
    for start in range(0, repetitions, block):
        stop = min(start+block, repetitions)
        indices = rng.integers(0, len(values), size=(stop-start, len(values)))
        result[start:stop] = values[indices].mean(axis=1)
    return result


def classify(metrics: dict[str, Any]) -> tuple[str, str, list[str]]:
    required = ['uniform_mean', 'suffix1_lower', 'suffix8_lower']
    if any(not np.isfinite(metrics[k]) for k in required):
        return 'INCONCLUSIVE', 'ENGINEERING_ABORT', ['nonfinite_metric']
    thresholds = [.20, .05, .02]
    failed = [key for key, threshold in zip(required, thresholds) if metrics[key] < threshold]
    if failed:
        return 'INCONCLUSIVE', 'TASK_INFORMATION_FAILURE', failed
    return 'SUPPORTED', 'TASK_INFORMATION_ONLY', []


def run(config_path: Path, output_override: Path | None = None) -> Path:
    started, cpu_started = time.monotonic(), time.process_time()
    c = json.loads(config_path.read_text())
    validate_config(c)
    status = git_output('status', '--porcelain')
    if status:
        raise RuntimeError('Formal screen requires a clean committed source checkout')
    validation_path = Path('artifacts/validation/suite.json')
    validation = json.loads(validation_path.read_text())
    sources = source_fingerprints()
    if not validation.get('pass') or validation.get('source_sha256') != sources:
        raise RuntimeError('Passing validation must match all current source hashes')
    output = output_override or Path(c['output'])
    output.mkdir(parents=True, exist_ok=False)
    frozen = [config_path, Path(c['protocol']), Path(c['entry_contract']), Path('uv.lock')]
    manifest = dict(experiment_id='EXP-0015', version='v1', workflow='orbit_information',
                    config=c, config_sha256=sha(config_path), status='RUNNING',
                    source_repo=str(Path.cwd()), git_commit=git_output('rev-parse', 'HEAD'),
                    git_status=status, source_sha256=sources,
                    frozen_hashes={str(p): sha(p) for p in frozen},
                    validation_sha256=sha(validation_path),
                    python=sys.version, executable=str(Path(sys.executable).resolve()),
                    executable_sha256=sha(Path(sys.executable).resolve()),
                    numpy=np.__version__, platform=platform.platform(), machine=platform.machine(),
                    research_updates=0, model_instances=0, gate1_claim=False,
                    planned_research_examples=c['task']['examples'],
                    scope='TASK_INFORMATION_ONLY_NOT_CONTROL_HEADROOM')
    write_json(output/'manifest.json', manifest)
    try:
        raw = generate(c['task']['examples'], c['task']['length'], c['task']['root'])
        with (output/'raw.npz').open('xb') as stream:
            np.savez_compressed(stream, **raw)
        manifest['raw_sha256'] = sha(output/'raw.npz')
        manifest['arrays'] = {key: dict(shape=list(value.shape), dtype=str(value.dtype), sha256=array_sha(value))
                              for key, value in raw.items()}
        write_json(output/'manifest.json', manifest)
        full = posterior(raw['y'])
        suffix1, suffix8 = posterior(raw['y'][:, -1:]), posterior(raw['y'][:, -8:])
        a = kl(full, np.full_like(full, 1/16))
        b, d = kl(full, suffix1), kl(full, suffix8)
        means = bootstrap_means(np.column_stack([b, d]), 77100007, 10000)
        data = dict(full=full, suffix1=suffix1, suffix8=suffix8,
                    uniform_kl=a, suffix1_kl=b, suffix8_kl=d, bootstrap_means=means)
        with (output/'oracle.npz').open('xb') as stream:
            np.savez_compressed(stream, **data)
        manifest.update(oracle_sha256=sha(output/'oracle.npz'),
                        oracle_arrays={key: dict(shape=list(value.shape), dtype=str(value.dtype), sha256=array_sha(value))
                                       for key, value in data.items()},
                        status='COMPLETED', actual_research_examples=len(raw['y']),
                        wall_seconds=time.monotonic()-started, cpu_seconds=time.process_time()-cpu_started)
        if manifest['wall_seconds'] > c['resource']['worker_wall_seconds_cap']:
            raise TimeoutError('Registered worker budget exhausted before analysis')
        write_json(output/'manifest.json', manifest)
        print(json.dumps(dict(status=manifest['status'], seconds=manifest['wall_seconds'], updates=0)), flush=True)
    except BaseException as error:
        manifest.update(status='FAILED', error=repr(error), wall_seconds=time.monotonic()-started)
        write_json(output/'manifest.json', manifest)
        raise
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    run(args.config, args.output)


if __name__ == '__main__':
    main()
