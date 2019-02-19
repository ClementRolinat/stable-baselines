from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from builtins import range
from future import standard_library
from future.utils import native_str
standard_library.install_aliases()
import pytest
import numpy as np

from stable_baselines.a2c import A2C
from stable_baselines.ppo1 import PPO1
from stable_baselines.ppo2 import PPO2
from stable_baselines.trpo_mpi import TRPO
from stable_baselines.common.identity_env import IdentityEnvMultiBinary, IdentityEnvMultiDiscrete
from stable_baselines.common.vec_env.dummy_vec_env import DummyVecEnv

MODEL_LIST = [
    A2C,
    PPO1,
    PPO2,
    TRPO
]


@pytest.mark.slow
@pytest.mark.parametrize("model_class", MODEL_LIST)
def test_identity_multidiscrete(model_class):
    """
    Test if the algorithm (with a given policy)
    can learn an identity transformation (i.e. return observation as an action)
    with a multidiscrete action space

    :param model_class: (BaseRLModel) A RL Model
    """
    env = DummyVecEnv([lambda: IdentityEnvMultiDiscrete(10)])

    model = model_class(native_str("MlpPolicy"), env)
    model.learn(total_timesteps=1000, seed=0)

    n_trials = 1000
    reward_sum = 0
    obs = env.reset()
    for _ in range(n_trials):
        action, _ = model.predict(obs)
        obs, reward, _, _ = env.step(action)
        reward_sum += reward

    assert np.array(model.action_probability(obs)).shape == (2, 1, 10), \
        "Error: action_probability not returning correct shape"
    assert np.prod(model.action_probability(obs, actions=env.action_space.sample()).shape) == 1, \
        "Error: not scalar probability"


@pytest.mark.slow
@pytest.mark.parametrize("model_class", MODEL_LIST)
def test_identity_multibinary(model_class):
    """
    Test if the algorithm (with a given policy)
    can learn an identity transformation (i.e. return observation as an action)
    with a multibinary action space

    :param model_class: (BaseRLModel) A RL Model
    """
    env = DummyVecEnv([lambda: IdentityEnvMultiBinary(10)])

    model = model_class(native_str("MlpPolicy"), env)
    model.learn(total_timesteps=1000, seed=0)

    n_trials = 1000
    reward_sum = 0
    obs = env.reset()
    for _ in range(n_trials):
        action, _ = model.predict(obs)
        obs, reward, _, _ = env.step(action)
        reward_sum += reward

    assert model.action_probability(obs).shape == (1, 10), \
        "Error: action_probability not returning correct shape"
    assert np.prod(model.action_probability(obs, actions=env.action_space.sample()).shape) == 1, \
        "Error: not scalar probability"
