from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from future import standard_library
standard_library.install_aliases()
def _assert_eq(left, right):
    assert left == right, '{} != {}'.format(left, right)


def _assert_neq(left, right):
    assert left != right, '{} == {}'.format(left, right)
