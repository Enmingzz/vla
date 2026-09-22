"""CPU checks of the scientific boundaries and action/temporal mapping."""
from pathlib import Path

import numpy as np
import pytest

from frequency_vla.robocasa_protocol import check_chunk, episode_seed, load_config, align_teacher_latent
from frequency_vla.robocasa_env import observation, STATE_KEYS, IMAGE_KEYS


def config():
    return load_config(Path(__file__).parents[1]/'configs/robocasa365_opsd.yaml')


def test_train_and_eval_seed_spaces_are_disjoint_and_h_independent():
    c = config()
    evaluation = {episode_seed(c,t,i) for t in range(10) for i in range(10)}
    training = {episode_seed(c,t,i,True) for t in range(10) for i in range(10000)}
    assert len(evaluation)==100 and not evaluation.intersection(training)
    assert episode_seed(c,4,7)==episode_seed(c,4,7)


@pytest.mark.parametrize('shape',[(20,12),(50,7),(49,12),(50,32)])
def test_reject_wrong_native_or_physical_action_shape(shape):
    with pytest.raises(ValueError):
        check_chunk(np.zeros(shape),config())


def test_action_chunk_and_finite_values():
    a = np.zeros((50,12))
    assert check_chunk(a,config()) is a
    a[0,11] = np.nan
    with pytest.raises(ValueError):
        check_chunk(a,config())


def test_teacher_shift_preserves_all_physical_and_padded_dimensions():
    student = np.arange(2*50*32).reshape(2,50,32)
    auxiliary = student+100000
    for offset in [0,5,10,15]:
        aligned = align_teacher_latent(student,auxiliary,offset,np)
        np.testing.assert_array_equal(aligned[:,:50-offset],student[:,offset:])
        if offset:
            np.testing.assert_array_equal(aligned[:,-offset:],auxiliary[:,-offset:])
        np.testing.assert_array_equal(aligned[:,:5,:12],student[:,offset:offset+5,:12])


def test_native_three_camera_preprocessing_and_state_order():
    image = np.arange(224*224*3,dtype=np.uint8).reshape(224,224,3)
    obs = {v:image.copy() for v in IMAGE_KEYS.values()}
    dims = [3,4,3,4,2]
    obs.update({k:np.full(d,i,dtype=np.float32) for i,(k,d) in enumerate(zip(STATE_KEYS,dims))})
    obs['annotation.human.task_description']='open the drawer'
    result = observation(obs)
    for key in IMAGE_KEYS:
        np.testing.assert_array_equal(result[key],image)
    np.testing.assert_array_equal(result['observation/state'],np.repeat(np.arange(5),dims))
    assert result['prompt']=='open the drawer'


def test_fresh_views_and_terminal_action_mask_count():
    from frequency_vla.robocasa_train import execute_block
    class Simulator:
        steps,done,success,identity = 0,False,False,{}
        def get_observation(self):
            return {'step':self.steps}
        def step(self,action):
            assert not self.done
            assert action[0]==self.steps
            self.steps += 1
            self.done = self.success = self.steps==7
    actions = np.repeat(np.arange(50)[:,None],12,axis=1)
    views,valid,identity,last = execute_block(Simulator(),actions)
    assert [o['step'] for o in views]==[0,5,7,7]
    assert valid==7 and identity['end_step']==7 and identity['episode_ended']
    assert last['step']==7


def test_paired_analysis_rejects_different_resets_and_retains_negative_change():
    from frequency_vla.robocasa_analysis import paired
    base = {'task_id':0,'episode_index':0,'episode_seed':27,'success':True,
        **{k:'same' for k in ['initial_state_sha256','initial_xml_sha256',
            'initial_observation_sha256','environment_metadata_sha256','config_sha256',
            'prediction_horizon','flow_steps','renderer']}}
    worse = dict(base,success=False)
    assert paired([base],[worse])['difference_b_minus_a']==-1.0
    with pytest.raises(ValueError,match='initial_state_sha256'):
        paired([base],[dict(worse,initial_state_sha256='different')])


def test_pairing_ignores_only_inferred_obj_mime(tmp_path):
    import hashlib
    from frequency_vla.logging_utils import write_json
    from frequency_vla.robocasa_env import Episode, UnpairedResetError
    from frequency_vla.robocasa_pairing import xml_comparison_hash
    xml = '<mujoco><asset><mesh file="m.obj" content_type="model/obj" scale="1 1 1"/></asset></mujoco>'
    other = xml.replace(' content_type="model/obj"','')
    assert xml_comparison_hash(xml) == xml_comparison_hash(other)
    for changed in [other.replace('m.obj','other.obj'),other.replace('1 1 1','1 1 2'),
                    xml.replace('m.obj','m.stl')]:
        assert xml_comparison_hash(xml) != xml_comparison_hash(changed)
    ep = Episode.__new__(Episode)
    ep.name,ep.index,ep.xml = 'Task',0,other
    identity = {'training':False,'initial_observation_sha256':'same-image',
                'initial_xml_sha256':hashlib.sha256(xml.encode()).hexdigest()}
    folder = tmp_path/'Task/episode_000'
    write_json(folder/'identity.json',identity)
    (folder/'model.xml').write_text(xml)
    ep.identity = dict(identity,initial_xml_sha256=hashlib.sha256(other.encode()).hexdigest())
    with pytest.raises(UnpairedResetError):
        ep.pair(tmp_path)
    ep.pair(tmp_path,allow_obj_mime_equivalence=True)
    assert ep.identity['initial_xml_sha256'] != identity['initial_xml_sha256']
    assert ep.identity['initial_xml_comparison_sha256'] == xml_comparison_hash(xml)
    ep.identity['initial_observation_sha256'] = 'different-image'
    with pytest.raises(UnpairedResetError,match='initial_observation_sha256'):
        ep.pair(tmp_path,allow_obj_mime_equivalence=True)


def test_cached_inference_allows_only_snapshot_phase_alias():
    import copy
    from frequency_vla.robocasa_pairing import same_inference_spec
    a = {'flow_steps':10,'sources':{'native':'unchanged'},'temporal_opsd':{
        'phase':'student','optimizer_step':500,'checkpoint':{'path':'step_500','manifest_sha256':'abc'}}}
    b = copy.deepcopy(a)
    b['temporal_opsd']['phase'] = 'step_500'
    assert same_inference_spec(a,b)
    for changed in [dict(b,flow_steps=9),dict(b,sources={'native':'different'})]:
        assert not same_inference_spec(a,changed)
    b['temporal_opsd']['checkpoint']['manifest_sha256'] = 'different'
    assert not same_inference_spec(a,b)


def test_analysis_reports_partial_comparison_on_common_episodes(tmp_path):
    from frequency_vla.robocasa_analysis import analyze
    from frequency_vla.logging_utils import append_record
    import json
    base = {'task_id':0,'episode_index':0,'episode_seed':27,'success':True,
        **{k:'same' for k in ['initial_state_sha256','initial_xml_sha256',
            'initial_observation_sha256','environment_metadata_sha256','config_sha256',
            'prediction_horizon','flow_steps','renderer']}}
    append_record(tmp_path/'h20/raw/original_h20/Task.jsonl',base)
    append_record(tmp_path/'h20/raw/original_h20/Task.jsonl',dict(base,episode_index=1))
    append_record(tmp_path/'train500/raw/step500_h20/Task.jsonl',dict(base,success=False))
    analyze(tmp_path)
    result = json.loads((tmp_path/'aggregated/progress.json').read_text())
    comparison = result['partial_comparisons']['opsd_recovery']
    assert comparison['matched_episodes'] == 1
    assert comparison['successes_a'] == 1 and comparison['successes_b'] == 0
    assert not result['conditions']['step500_h20']['complete']


def test_recorded_counter_regions_remove_unordered_left_right_swap():
    from frequency_vla.robocasa_reset import relabel_regions
    left = {'offset':[-1.3,0.,0.46],'size':[1.6,.65]}
    right = {'offset':[1.2,0.,0.46],'size':[1.9,.65]}
    saved = [dict(right,name='geom_0'),dict(left,name='geom_1')]
    a = relabel_regions({'geom_0':left,'geom_1':right},saved)
    b = relabel_regions({'geom_0':right,'geom_1':left},saved)
    assert a == b == {'geom_0':right,'geom_1':left}
    # The same RNG-selected index now points to the same geometry. No RNG draw
    # or placement is added, and native region dictionaries are retained.
    assert a['geom_0'] is right and a['geom_1'] is left
    assert relabel_regions({'geom_0':left,'geom_1':right},saved[:1]) == b


def test_recorded_regions_never_replace_geometry_or_guess_missing_order():
    from frequency_vla.robocasa_reset import relabel_regions
    regions = {'geom_0':{'size':[1.,1.],'offset':[0.,0.,0.]},
               'geom_1':{'size':[2.,2.],'offset':[1.,0.,0.]},
               'geom_2':{'size':[3.,3.],'offset':[2.,0.,0.]}}
    assert relabel_regions(regions,[dict(regions['geom_2'],name='geom_0')]) is regions
    bad = {'name':'geom_0','size':[99.,99.],'offset':[0.,0.,0.]}
    assert relabel_regions(regions,[bad]) is regions
