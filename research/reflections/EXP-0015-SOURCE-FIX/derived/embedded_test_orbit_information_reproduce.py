"""Provenance, failure-receipt and full-verdict regression tests without task samples."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import neurotopo.orbit_information_reproduce as audit


@pytest.fixture
def frozen_source(tmp_path):
    source=tmp_path/'source';source.mkdir()
    files={'src/neurotopo/orbit_information_reproduce.py':Path(audit.__file__).read_text(),
           'src/neurotopo/sentinel.py':'value=1\n','configs/exp0015_v1.json':'{"test_fixture":true}\n',
           'experiments/EXP-0015/v1.md':'frozen protocol\n','research/entries/EXP-0015.json':'{}\n','uv.lock':'fixture lock\n'}
    for name,text in files.items():
        path=source/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(text)
    fingerprints={str(p.relative_to(source)):audit.file_hash(p) for p in sorted((source/'src').rglob('*.py'))}
    validation=source/'artifacts/validation/suite.json';validation.parent.mkdir(parents=True)
    validation.write_text(json.dumps(dict(pass_=True)))
    validation.write_text(json.dumps({'pass':True,'source_sha256':fingerprints}))
    subprocess.run(['git','init','-q',str(source)],check=True)
    subprocess.run(['git','-C',str(source),'add','.'],check=True)
    subprocess.run(['git','-C',str(source),'-c','user.name=Engineering Test','-c','user.email=fixture@example.invalid','commit','-qm','Frozen test fixture'],check=True)
    commit=subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()
    manifest=dict(git_commit=commit,source_sha256=fingerprints,
                  frozen_hashes={name:audit.file_hash(source/name) for name in files if not name.startswith('src/')},
                  config_sha256=audit.file_hash(source/'configs/exp0015_v1.json'),config={'test_fixture':True},
                  validation_sha256=audit.file_hash(validation),python=sys.version,numpy=np.__version__,
                  executable_sha256=audit.file_hash(Path(sys.executable).resolve()))
    return source,manifest


def test_frozen_provenance_passes(frozen_source):
    source,manifest=frozen_source
    assert audit.verify_provenance(manifest,source)['source_commit']==manifest['git_commit']


@pytest.mark.parametrize('field',['config_sha256','validation_sha256','git_commit','executable_sha256','numpy','source_sha256','frozen_hashes'])
def test_changed_provenance_declarations_fail(frozen_source,field):
    source,manifest=frozen_source;bad=deepcopy(manifest)
    if isinstance(bad[field],dict):
        key=next(iter(bad[field]));bad[field][key]='0'*64
    else:bad[field]='invalid'
    with pytest.raises((ValueError,subprocess.CalledProcessError)):
        audit.verify_provenance(bad,source)


@pytest.mark.parametrize('relative',['configs/exp0015_v1.json','experiments/EXP-0015/v1.md','research/entries/EXP-0015.json','uv.lock','src/neurotopo/sentinel.py','artifacts/validation/suite.json'])
def test_changed_frozen_file_fails(frozen_source,relative):
    source,manifest=frozen_source;(source/relative).write_text('changed bytes\n')
    with pytest.raises(ValueError):audit.verify_provenance(manifest,source)


def result_fixture():
    metrics=dict(uniform_mean=.3,suffix1_mean=.1,suffix8_mean=.001,suffix1_lower=.09,suffix1_upper=.11,suffix8_lower=.0009,suffix8_upper=.0011)
    analysis=dict(metrics=deepcopy(metrics),failed=['suffix8_lower'],substatus='TASK_INFORMATION_FAILURE',decision='INCONCLUSIVE',gate1_claim=False,research_updates=0)
    return metrics,analysis


@pytest.mark.parametrize('field',['suffix1_upper','suffix8_upper','suffix1_mean','suffix8_mean','decision'])
def test_all_interval_fields_and_top_level_decision_are_audited(field):
    metrics,analysis=result_fixture();assert max(audit.verify_result(metrics,analysis).values())==0
    if field=='decision':analysis[field]='SUPPORTED'
    else:analysis['metrics'][field]+=.1
    with pytest.raises(ValueError):audit.verify_result(metrics,analysis)


@pytest.mark.parametrize('error',[TimeoutError('injected deadline'),ValueError('injected hash mismatch')])
def test_reproduction_failure_is_saved_and_alarm_disarmed(tmp_path,monkeypatch,error):
    armed=[]
    monkeypatch.setattr(audit.signal,'setitimer',lambda kind,seconds:armed.append(seconds))
    def fail(*args):raise error
    monkeypatch.setattr(audit,'evaluate',fail)
    output=tmp_path/'receipt.json';report=audit.execute(tmp_path,output)
    assert armed==[300,0]
    assert not report['pass_'] and report['substatus']=='REPRODUCTION_UNRESOLVED'
    assert json.loads(output.read_text())['error']==repr(error)
    with pytest.raises(FileExistsError):audit.execute(tmp_path,output)
