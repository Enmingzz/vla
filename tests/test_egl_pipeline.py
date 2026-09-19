"""Guard the fixed EGL matrix, success-only timings and fresh-training schedule."""
import json
from pathlib import Path
import sys
import types

import pytest

from frequency_vla.config import load_config
from frequency_vla.egl_pipeline import validate_plan, summarize_condition, execute
from frequency_vla.logging_utils import digest, write_json


def test_common_egl_matrix_requires_all_five_conditions():
    plan=load_config(Path(__file__).resolve().parents[1]/'configs/egl_pipeline.yaml')
    validate_plan(plan)
    assert len(plan['conditions'])*10*plan['episodes_per_task']==500
    for field,value in [('workers',8),('renderer','osmesa'),('optimizer_updates_added',0)]:
        with pytest.raises(ValueError,match='protocol changed'):
            validate_plan(dict(plan,**{field:value}))


def test_successful_time_excludes_failures_and_handles_no_success():
    rows=[dict(prediction_horizon=50,native_prediction_horizon=10,replan_steps=20,success=True,policy_calls=2,
        controlled_environment_steps=25,environment_steps=35,average_executed_actions_per_policy_call=12.5,wall_clock_seconds=2.),
        dict(prediction_horizon=50,native_prediction_horizon=10,replan_steps=20,success=False,policy_calls=26,
        controlled_environment_steps=520,environment_steps=530,average_executed_actions_per_policy_call=20,wall_clock_seconds=50.)]
    result=summarize_condition(dict(step=1500,horizon=20),rows)
    assert result['mean_wall_clock_seconds']==26 and result['mean_successful_seconds']==2
    assert result['mean_successful_actions']==25
    assert summarize_condition(dict(step=1500,horizon=20),rows[1:])['mean_successful_seconds'] is None


def test_cpu_guard_prevents_running_changed_protocol(tmp_path,monkeypatch):
    # Guard fails before any connection or subprocess; no synthetic research outputs.
    plan=load_config(Path(__file__).resolve().parents[1]/'configs/egl_pipeline.yaml')
    write_json(tmp_path/'provenance/split_audit.json',dict(plan_sha256=digest(plan)))
    write_json(tmp_path/'provenance/preflight_checks.json',dict(passed=False,sources={}))
    module=types.ModuleType('openpi_client.websocket_client_policy')
    module.WebsocketClientPolicy=lambda *a,**k:pytest.fail('Server contacted before CPU checks')
    monkeypatch.setitem(sys.modules,'openpi_client.websocket_client_policy',module)
    with pytest.raises(ValueError,match='CPU checks'):
        execute(tmp_path,plan,12345,tmp_path/'checkpoints')


def test_fresh_schedule_trains_exact_milestones_without_resuming(tmp_path, monkeypatch):
    from frequency_vla import egl_pipeline as module
    project=Path(__file__).resolve().parents[1]
    plan=load_config(project/'configs/egl_pipeline.yaml')
    config=load_config(project/plan['training_config'])
    write_json(tmp_path/'provenance/split_audit.json',dict(plan_sha256=digest(plan),
        training_config_sha256=digest(config),base_checkpoint_manifest_sha256='SYNTHETIC_BASE'))
    write_json(tmp_path/'provenance/preflight_checks.json',dict(passed=True,sources={}))
    write_json(tmp_path/'provenance/training_setup.json',dict(config=config,
        base_experiment_spec=dict(checkpoint_object_manifest_sha256='SYNTHETIC_BASE')))
    (tmp_path/'logs').mkdir()
    for key,value in dict(MUJOCO_GL='egl',PYOPENGL_PLATFORM='egl',
                          FREQUENCY_CONFIG=str(project/'configs/prediction50_egl.yaml')).items():
        monkeypatch.setenv(key,value)
    # Mock process boundaries only; this tests orchestration, not research outcomes.
    monkeypatch.setattr(module,'validate_records',lambda rows:None)
    calls=[]
    def run(command,**kwargs):
        calls.append(command)
        def value(flag):
            return command[command.index(flag)+1]
        if 'tests/check_parallel_osmesa.py' in command:
            write_json(tmp_path/'provenance/render_context_check.json',dict(passed=True,renderer=dict(backend='egl')))
        elif '--stage' in command and value('--stage')=='diagnostic':
            write_json(tmp_path/'provenance/diagnostic.json',dict(passed=True,optimizer_steps=0))
        elif 'frequency_vla.opsd_evaluate' in command:
            seed,episodes=int(value('--seed')),int(value('--episodes'))
            tasks=[int(t) for t in command[command.index('--task-ids')+1:]] if '--task-ids' in command else range(10)
            folder=Path(value('--results-dir'))/'raw/smoke/libero_10'/('seed_'+str(seed))/('H_'+value('--horizon'))
            folder.mkdir(parents=True)
            for task in tasks:
                (folder/('task_'+str(task)+'.jsonl')).write_text('\n'.join(json.dumps(dict(seed=seed,task_id=task,episode_index=e)) for e in range(episodes)))
                write_json(folder/('task_'+str(task)+'.manifest.json'),dict(status='complete'))
    monkeypatch.setattr(module.subprocess,'run',run)
    execute(tmp_path,plan,12345,tmp_path/'checkpoints')
    stages=[c[c.index('--stage')+1] for c in calls if '--stage' in c]
    assert stages==['diagnostic','snapshot','snapshot','snapshot','snapshot',
                    'train','save','snapshot','train','save','snapshot','train','save','snapshot','status']
    assert [int(c[c.index('--end-step')+1]) for c in calls if '--end-step' in c]==[500,1000,1500]
    evaluations=[c for c in calls if 'frequency_vla.opsd_evaluate' in c]
    assert len(evaluations)==7  # Two runtime pilots and five formal conditions.
    assert all(c[c.index('--workers')+1]=='4' for c in evaluations[2:])
    assert json.loads((tmp_path/'provenance/study_complete.json').read_text())['optimizer_updates_added']==1500


def test_complete_five_condition_analysis_keeps_one_runtime_and_success_timings(tmp_path, monkeypatch):
    import math
    from frequency_vla import egl_pipeline as module
    plan=load_config(Path(__file__).resolve().parents[1]/'configs/egl_pipeline.yaml')
    config=load_config(Path(__file__).resolve().parents[1]/'configs/opsd_egl_1500.yaml')
    checkpoints={}
    for step in [500,1000,1500]:
        checkpoint=tmp_path/'checkpoints'/('step_'+str(step))
        manifest=dict(step=step,config=config,base_object_manifest_sha256='SYNTHETIC_BASE',
                      reloaded_native_inference_max_abs_difference=0,resumed_from=None,files={})
        write_json(checkpoint/'training_manifest.json',manifest)
        checkpoints[str(step)]=dict(path=str(checkpoint),manifest_sha256=digest(manifest))
        write_json(tmp_path/'provenance'/('step_'+str(step)+'.json'),checkpoints[str(step)])
    write_json(tmp_path/'provenance/training_runtime.json',dict(renderer=dict(backend='egl'),official_start_step=0))
    pool=[dict(task_id=t,initial_state_index=10,initial_state_sha256='SYNTHETIC_TRAIN_'+str(t)) for t in range(4)]
    (tmp_path/'training.jsonl').write_text('\n'.join(json.dumps(dict(optimizer_step=i,diagnostic=False,student_behavior_version=i-1)) for i in range(1,1501)))
    (tmp_path/'rollouts.jsonl').write_text('\n'.join(json.dumps(dict(optimizer_step=i,diagnostic=False,initial_states=pool)) for i in range(1,1501)))
    states=[dict(task_id=t,initial_state_index=30+e,initial_state_sha256=str((t,e))) for t in range(10) for e in range(10)]
    write_json(tmp_path/'provenance/study_plan.json',plan)
    write_json(tmp_path/'provenance/split_audit.json',dict(plan_sha256=digest(plan),training_config_sha256=digest(config),base_checkpoint_manifest_sha256='SYNTHETIC_BASE',
        training_initial_states=pool,evaluation_initial_states=states))
    write_json(tmp_path/'provenance/study_complete.json',dict(complete=True,plan_sha256=digest(plan),
        completed_conditions=[module.condition_id(c) for c in plan['conditions']],optimizer_updates_added=1500))
    write_json(tmp_path/'provenance/training_setup.json',dict(config=config,sources={'SYNTHETIC':'SYNTHETIC'}))
    stages=[]
    for c in plan['conditions']:
        output=tmp_path/'evaluations'/('step_'+str(c['step']))
        folder=output/'raw/smoke/libero_10/seed_27'/('H_'+str(c['horizon']))
        folder.mkdir(parents=True)
        spec=dict(prediction_horizon=50,flow_steps=10,temporal_opsd=dict(optimizer_step=c['step'],
            checkpoint=checkpoints.get(str(c['step'])),sources={'SYNTHETIC':'SYNTHETIC'},evaluation_sampler='unchanged Pi0.sample_actions; dynamic parameter arguments'))
        server=dict(server_instance_id='SYNTHETIC_SINGLE_SERVER',experiment_spec=spec,inference_fingerprint=digest(spec),
                    evaluation_spec=dict(rendering=dict(backend='egl')))
        for t in range(10):
            rows=[]
            for e in range(10):
                success=e<5+c['step']//500
                steps=25 if success else 520
                calls=math.ceil(steps/c['horizon'])
                video='videos/H_{}/task_{}_episode_{}.mp4'.format(c['horizon'],t,e)
                write_json(output/video,dict(streams=[dict(width=224,height=224,r_frame_rate='10/1',nb_frames=str(steps))]))
                rows.append(dict(task_suite='libero_10',task_id=t,episode_index=e,seed=27,initial_state_index=30+e,
                    initial_state_sha256=str((t,e)),first_observation_sha256=str((t,e)),episode_rng_seed=e,
                    task_description='SYNTHETIC_TASK_'+str(t),prediction_horizon=50,native_prediction_horizon=10,replan_steps=c['horizon'],
                    inference_fingerprint=digest(spec),evaluation_fingerprint='SYNTHETIC_EGL',success=success,
                    controlled_environment_steps=steps,settling_steps=10,environment_steps=steps+10,policy_calls=calls,
                    policy_call_control_steps=list(range(0,steps,c['horizon'])),
                    average_executed_actions_per_policy_call=steps/calls,wall_clock_seconds=2. if success else 50.,video_path=video))
            (folder/('tasks_'+str(t)+'.jsonl')).write_text('\n'.join(json.dumps(r) for r in rows)+'\n')
            write_json(folder/('tasks_'+str(t)+'.manifest.json'),dict(status='complete',server=server))
        stages.append(dict(stage=module.condition_id(c),event='complete',workers=4,pilot=False,seconds=100.))
    (tmp_path/'stages.jsonl').write_text('\n'.join(json.dumps(r) for r in stages))
    monkeypatch.setattr(module.subprocess,'check_output',lambda command,**kwargs:Path(command[-1]).read_text())
    validation=module.analyze(tmp_path)
    assert validation['complete'] and validation['formal_episodes']==validation['validated_videos']==500
    assert validation['same_renderer']=='egl' and validation['optimizer_updates_added']==1500
    assert validation['training_renderer']==validation['evaluation_renderer']=='egl'
    assert '0 (Original)' in (tmp_path/'FINDINGS.md').read_text()
    assert 'Controlled training and evaluation' in (tmp_path/'FINDINGS.md').read_text()
    shard=tmp_path/'evaluations/step_1500/raw/smoke/libero_10/seed_27/H_20/tasks_0.jsonl'
    rows=[json.loads(line) for line in shard.read_text().splitlines()]
    rows[0]['first_observation_sha256']='CHANGED'
    shard.write_text('\n'.join(json.dumps(r) for r in rows))
    with pytest.raises(ValueError,match='Unpaired comparison'):
        module.analyze(tmp_path)
