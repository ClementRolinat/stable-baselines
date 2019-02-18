from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from builtins import super
from builtins import range
from builtins import dict
from future import standard_library
standard_library.install_aliases()
import os

import gym
import pytest
import tensorflow as tf

from stable_baselines import A2C, ACER, ACKTR, DQN, PPO1, PPO2, TRPO, SAC, DDPG
from stable_baselines.common.policies import FeedForwardPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.deepq.policies import FeedForwardPolicy as DQNPolicy
from stable_baselines.ddpg.policies import FeedForwardPolicy as DDPGPolicy
from stable_baselines.sac.policies import FeedForwardPolicy as SACPolicy

N_TRIALS = 100


class CustomCommonPolicy(FeedForwardPolicy):
    def __init__(self, *args, **kwargs):
        # Default value
        if 'net_arch' not in kwargs:
            kwargs['net_arch'] = [8, dict(vf=[8, 8], pi=[8, 8])]
        super(CustomCommonPolicy, self).__init__(*args, feature_extraction="mlp", **kwargs)


class CustomDQNPolicy(DQNPolicy):
    def __init__(self, *args, **kwargs):
        # Default value
        if 'layers' not in kwargs:
            kwargs['layers'] = [8, 8]
        super(CustomDQNPolicy, self).__init__(*args, feature_extraction="mlp", **kwargs)


class CustomDDPGPolicy(DDPGPolicy):
    def __init__(self, *args, **kwargs):
        # Default value
        if 'layers' not in kwargs:
            kwargs['layers'] = [8, 8]
        super(CustomDDPGPolicy, self).__init__(*args, feature_extraction="mlp", **kwargs)


class CustomSACPolicy(SACPolicy):
    def __init__(self, *args, **kwargs):
        # Default value
        if 'layers' not in kwargs:
            kwargs['layers'] = [8, 8]
        super(CustomSACPolicy, self).__init__(*args, feature_extraction="mlp", **kwargs)


# MODEL_CLASS, POLICY_CLASS, POLICY_KWARGS
MODEL_DICT = {
    'a2c': (A2C, CustomCommonPolicy, dict(act_fun=tf.nn.relu, net_arch=[12, dict(vf=[16], pi=[8])])),
    'acer': (ACER, CustomCommonPolicy, dict(act_fun=tf.nn.relu)),
    'acktr': (ACKTR, CustomCommonPolicy, dict(act_fun=tf.nn.relu)),
    'dqn': (DQN, CustomDQNPolicy, dict(layers=[4, 4], dueling=False)),
    'ddpg': (DDPG, CustomDDPGPolicy, dict(layers=[16, 16], layer_norm=False)),
    'ppo1': (PPO1, CustomCommonPolicy, dict(act_fun=tf.nn.relu, net_arch=[8, 4])),
    'ppo2': (PPO2, CustomCommonPolicy, dict(act_fun=tf.nn.relu, net_arch=[4, 4])),
    'sac': (SAC, CustomSACPolicy, dict(layers=[16, 16])),
    'trpo': (TRPO, CustomCommonPolicy, dict(act_fun=tf.nn.relu)),
}


@pytest.mark.parametrize("model_name", MODEL_DICT.keys())
def test_custom_policy(model_name):
    """
    Test if the algorithm (with a custom policy) can be loaded and saved without any issues.
    :param model_name: (str) A RL model
    """

    try:
        model_class, policy, _ = MODEL_DICT[model_name]
        env = 'MountainCarContinuous-v0' if model_name in ['ddpg', 'sac'] else 'CartPole-v1'

        # create and train
        model = model_class(policy, env)
        model.learn(total_timesteps=100, seed=0)

        env = model.get_env()
        # predict and measure the acc reward
        obs = env.reset()
        for _ in range(N_TRIALS):
            action, _ = model.predict(obs)
            # Test action probability method
            if model_name not in ['ddpg', 'sac']:
                model.action_probability(obs)
            obs, _, _, _ = env.step(action)
        # saving
        model.save("./test_model")
        del model, env
        # loading
        _ = model_class.load("./test_model", policy=policy)

    finally:
        if os.path.exists("./test_model"):
            os.remove("./test_model")


@pytest.mark.parametrize("model_name", MODEL_DICT.keys())
def test_custom_policy_kwargs(model_name):
    """
    Test if the algorithm (with a custom policy) can be loaded and saved without any issues.
    :param model_name: (str) A RL model
    """

    try:
        model_class, policy, policy_kwargs = MODEL_DICT[model_name]
        env = 'MountainCarContinuous-v0' if model_name in ['ddpg', 'sac'] else 'CartPole-v1'

        # Should raise an error when a wrong keyword is passed
        with pytest.raises(ValueError):
            model_class(policy, env, policy_kwargs=dict(this_throws_error='maybe'))

        # create and train
        model = model_class(policy, env, policy_kwargs=policy_kwargs)
        model.learn(total_timesteps=100, seed=0)

        model.save("./test_model")
        del model

        # loading

        env = DummyVecEnv([lambda: gym.make(env)])

        # Load with specifying policy_kwargs
        model = model_class.load("./test_model", policy=policy, env=env, policy_kwargs=policy_kwargs)
        model.learn(total_timesteps=100, seed=0)
        del model

        # Load without specifying policy_kwargs
        model = model_class.load("./test_model", policy=policy, env=env)
        model.learn(total_timesteps=100, seed=0)
        del model

        # Load with different wrong policy_kwargs
        with pytest.raises(ValueError):
            _ = model_class.load("./test_model", policy=policy, env=env, policy_kwargs=dict(wrong="kwargs"))

    finally:
        if os.path.exists("./test_model"):
            os.remove("./test_model")
