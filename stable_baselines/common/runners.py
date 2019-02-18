from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from builtins import range
from future import standard_library
standard_library.install_aliases()
import numpy as np
from abc import ABCMeta, abstractmethod


class AbstractEnvRunner(ABCMeta):
    def __init__(self, **_3to2kwargs):
        n_steps = _3to2kwargs['n_steps']; del _3to2kwargs['n_steps']
        model = _3to2kwargs['model']; del _3to2kwargs['model']
        env = _3to2kwargs['env']; del _3to2kwargs['env']
        """
        A runner to learn the policy of an environment for a model

        :param env: (Gym environment) The environment to learn from
        :param model: (Model) The model to learn
        :param n_steps: (int) The number of steps to run for each environment
        """
        self.env = env
        self.model = model
        n_env = env.num_envs
        self.batch_ob_shape = (n_env*n_steps,) + env.observation_space.shape
        self.obs = np.zeros((n_env,) + env.observation_space.shape, dtype=env.observation_space.dtype.name)
        self.obs[:] = env.reset()
        self.n_steps = n_steps
        self.states = model.initial_state
        self.dones = [False for _ in range(n_env)]

    @abstractmethod
    def run(self):
        """
        Run a learning step of the model
        """
        raise NotImplementedError
