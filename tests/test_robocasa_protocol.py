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
