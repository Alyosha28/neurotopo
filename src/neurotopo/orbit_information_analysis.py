"""Read-only analysis of EXP-0015; no new samples or optimizer updates."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from .orbit_information import classify, sha, validate_config
from .run import write_json


def analyze(run: Path) -> dict:
    if (run/'analysis.json').exists():
        raise FileExistsError('Refusing to overwrite analysis')
    failure = run/'worker_failure.json'
    if failure.exists():
        incident = json.loads(failure.read_text())
        result = dict(experiment_id='EXP-0015', decision='INCONCLUSIVE',
                      substatus=incident['substatus'], gate1_claim=False, research_updates=0,
                      failure=incident, interpretation='No scientific gate verdict from incomplete execution')
    else:
        manifest = json.loads((run/'manifest.json').read_text())
        validate_config(manifest['config'])
        if manifest['status'] != 'COMPLETED' or manifest['research_updates'] != 0:
            raise ValueError('Incomplete or non-zero-update screen')
        for file, key in [('raw.npz','raw_sha256'), ('oracle.npz','oracle_sha256')]:
            if sha(run/file) != manifest[key]:
                raise ValueError(f'Artifact SHA differs: {file}')
        with np.load(run/'oracle.npz', allow_pickle=False) as arrays:
            for key in arrays.files:
                if not np.isfinite(arrays[key]).all():
                    raise ValueError('Nonfinite saved oracle output')
            intervals = np.quantile(arrays['bootstrap_means'], [.025, .975], axis=0, method='linear')
            metrics = dict(uniform_mean=float(arrays['uniform_kl'].mean()),
                           suffix1_mean=float(arrays['suffix1_kl'].mean()),
                           suffix8_mean=float(arrays['suffix8_kl'].mean()),
                           suffix1_lower=float(intervals[0,0]), suffix1_upper=float(intervals[1,0]),
                           suffix8_lower=float(intervals[0,1]), suffix8_upper=float(intervals[1,1]))
        decision, substatus, failed = classify(metrics)
        result = dict(experiment_id='EXP-0015', decision=decision, substatus=substatus,
                      metrics=metrics, failed=failed, gate1_claim=False, research_updates=0,
                      research_examples=manifest['actual_research_examples'],
                      independent_review='PENDING',
                      interpretation='Fixed task information only; not model headroom, organization effect, or Gate1',
                      next_action='Close fixed ORBIT-SOFT task configuration; no noise/length search' if failed
                      else 'Keep eligibility and control-qualification HOLD; no automatic training')
    write_json(run/'analysis.json', result)
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return result


def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument('--run', type=Path, required=True)
    analyze(parser.parse_args().run)


if __name__ == '__main__':
    main()
