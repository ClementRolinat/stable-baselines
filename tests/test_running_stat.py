from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from builtins import range
from future import standard_library
standard_library.install_aliases()
import numpy as np

from stable_baselines.common.running_stat import RunningStat


def test_running_stat():
    """
    test RunningStat object
    """
    for shape in ((), (3,), (3, 4)):
        hist = []
        running_stat = RunningStat(shape)
        for _ in range(5):
            val = np.random.randn(*shape)
            running_stat.push(val)
            hist.append(val)
            _mean = np.mean(hist, axis=0)
            assert np.allclose(running_stat.mean, _mean)
            _var = np.square(_mean) if (len(hist) == 1) else np.var(hist, ddof=1, axis=0)
            assert np.allclose(running_stat.var, _var)
