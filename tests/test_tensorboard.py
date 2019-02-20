from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from future import standard_library
from future.utils import native_str
standard_library.install_aliases()
import os
import shutil

import pytest

from stable_baselines import A2C, ACER, ACKTR, DQN, DDPG, PPO1, PPO2, SAC, TRPO

TENSORBOARD_DIR = '/tmp/tb_dir/'

if os.path.isdir(TENSORBOARD_DIR):
    shutil.rmtree(TENSORBOARD_DIR)

MODEL_DICT = {
    'a2c': (A2C, native_str('CartPole-v1')),
    'acer': (ACER, native_str('CartPole-v1')),
    'acktr': (ACKTR, native_str('CartPole-v1')),
    'dqn': (DQN, native_str('CartPole-v1')),
    'ddpg': (DDPG, native_str('Pendulum-v0')),
    'ppo1': (PPO1, native_str('CartPole-v1')),
    'ppo2': (PPO2, native_str('CartPole-v1')),
    'sac': (SAC, native_str('Pendulum-v0')),
    'trpo': (TRPO, native_str('CartPole-v1')),
}

N_STEPS = 1000


@pytest.mark.parametrize("model_name", MODEL_DICT.keys())
def test_tensorboard(model_name):
    algo, env_id = MODEL_DICT[model_name]
    model = algo(native_str('MlpPolicy'), env_id, verbose=1, tensorboard_log=TENSORBOARD_DIR)
    model.learn(N_STEPS)
    model.learn(N_STEPS, reset_num_timesteps=False)
