"""Reusing a reference must not rerun it or silently accept changed measurements."""
import json
from pathlib import Path

import pytest

from frequency_vla import study_runner
from frequency_vla.cached_reference import prepare_reference, reference_inventory, verify_reference
from frequency_vla.config import load_config
from frequency_vla.logging_utils import digest, write_json


def test_reference_reuse_pins_source_and_copy_bytes(tmp_path):
    archive, root = tmp_path / "archive", tmp_path / "new"
    checkpoint = dict(path="SYNTHETIC", manifest_sha256="manifest")
    write_json(archive / 'provenance/step_1000.json', checkpoint)
    write_json(archive / 'provenance/final_checks.json', dict(complete=True, all_checkpoint_file_checksums_verified=True))
    (archive / 'stages.jsonl').write_text(json.dumps(dict(stage='confirmation_libero_10_step1000_H20', event='complete', seconds=123)))
    folder = archive / 'evaluations/confirmation/step_1000/raw/smoke/libero_10/seed_27/H_20'
    folder.mkdir(parents=True)
    server = dict(server_instance_id='old', experiment_spec=dict(temporal_opsd=dict(optimizer_step=1000, checkpoint=checkpoint)),
                  evaluation_spec=dict(rendering=dict(backend='osmesa')))
    for t in range(10):
        (folder / ('task_{}.jsonl'.format(t))).write_text('{"synthetic": true}\n')
        write_json(folder / ('task_{}.manifest.json'.format(t)), dict(status='complete', server=server))
    plan = dict(resume_step=1000, comparison_step=1000, resume_manifest_sha256='manifest', renderer='osmesa',
        confirmation=dict(suite='libero_10', horizons={1000:[20],1500:[20]}, seed=27, initial_state_start=30, episodes_per_task=10),
        cached_reference=dict(archive=str(archive), step=1000, inventory_sha256=digest(reference_inventory(archive,1000))))
    prepared = prepare_reference(root, plan)
    assert prepared == verify_reference(root, plan)
    assert prepared['episodes'] == 100
    assert json.loads((root / 'stages.jsonl').read_text())['charged_to_current_allocation'] is False
    copied = next((root / 'evaluations').rglob('*.jsonl'))
    original = copied.read_text()
    copied.write_text('changed')
    with pytest.raises(ValueError, match='Copied reference'):
        verify_reference(root, plan)
    copied.write_text(original)
    source = next(folder.glob('*.jsonl'))
    source.write_text('changed')
    with pytest.raises(ValueError, match='archive changed'):
        verify_reference(root, plan)


def test_cached_runner_trains_exactly_500_then_evaluates_only_final(tmp_path, monkeypatch):
    project = Path(__file__).resolve().parents[1]
    plan_path = project / 'configs/autoresearch_round4.yaml'
    plan = load_config(plan_path)
    reference = dict(condition='confirmation_libero_10_step1000_H20')
    (tmp_path / 'logs').mkdir()
    write_json(tmp_path / 'provenance/split_audit.json', dict(plan_sha256=digest(plan), cached_reference=reference))
    write_json(tmp_path / 'provenance/preflight_checks.json', dict(passed=True, sources={}))
    write_json(tmp_path / 'provenance/parallel_environment_check.json', dict(passed=True, source_sha256={}))
    monkeypatch.setattr('frequency_vla.cached_reference.verify_reference', lambda root, plan: reference)
    operations = []
    def execute(command, **kwargs):
        if 'frequency_vla.opsd_evaluate' in command:
            operations.append('evaluate')
            assert 'step_1500' in command[command.index('--results-dir')+1]
            assert command[command.index('--episodes')+1] == '10'
            return
        operation = command[command.index('--stage')+1]
        operations.append(operation)
        if operation == 'train':
            assert command[command.index('--end-step')+1] == '1500'
        if operation == 'snapshot':
            assert command[command.index('--snapshot')+1] == '1500'
    monkeypatch.setattr(study_runner.subprocess, 'run', execute)
    monkeypatch.setattr(study_runner.sys, 'argv', ['study_runner', '--plan', str(plan_path),
        '--training-config', str(project / 'configs/opsd_continuation_1500.yaml'),
        '--parent-checkpoint', 'SYNTHETIC_STEP_1000', '--port', '12345',
        '--results-dir', str(tmp_path), '--checkpoint-root', str(tmp_path / 'checkpoints')])
    study_runner.main()
    assert operations == ['diagnostic', 'resume', 'train', 'save', 'status', 'snapshot', 'evaluate']
    completed = json.loads((tmp_path / 'provenance/study_complete.json').read_text())
    assert completed['optimizer_steps_added'] == completed['comparison_optimizer_steps_added'] == 500
    assert completed['reused_conditions'] == [reference['condition']]
    assert len(completed['completed_conditions']) == 2


def test_1500_config_changes_only_the_authorized_update_budget():
    project = Path(__file__).resolve().parents[1]
    before = load_config(project / 'configs/opsd_continuation_1000.yaml')
    after = load_config(project / 'configs/opsd_continuation_1500.yaml')
    assert {k for k in before if before[k] != after[k]} == {'optimizer_steps','checkpoint_steps'}
    assert after['optimizer_steps'] - before['optimizer_steps'] == 500
    assert after['checkpoint_steps'] == [1000,1500]
