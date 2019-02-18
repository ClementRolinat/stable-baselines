from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from future import standard_library
standard_library.install_aliases()
import pytest


def pytest_addoption(parser):
    parser.addoption("--rungpu", action="store_true", default=False, help="run gpu tests")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--rungpu"):
        return
    skip_gpu = pytest.mark.skip(reason="need --rungpu option to run")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip_gpu)
