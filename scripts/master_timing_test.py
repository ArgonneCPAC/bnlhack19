import sys
import numpy as np
import chopperhack19.mock_obs
from chopperhack19.mock_obs.tests import random_weighted_points
from chopperhack19.mock_obs.tests.generate_test_data import (
    DEFAULT_RBINS_SQUARED)
from time import time

if len(sys.argv) == 2:
    func_str = sys.argv[1]
else:
    func_str = 'count_weighted_pairs_3d_cpu'

func = getattr(chopperhack19.mock_obs, func_str)
if func is None:
    raise ImportError('could not import %s' % func_str)

Lbox = 1000.
result = np.zeros_like(DEFAULT_RBINS_SQUARED)

n1 = 128
n2 = 128
x1, y1, z1, w1 = random_weighted_points(n1, Lbox, 0)
x2, y2, z2, w2 = random_weighted_points(n2, Lbox, 1)

func(
    x1, y1, z1, w1, x2, y2, z2, w2, DEFAULT_RBINS_SQUARED, result)

n1 = 50013
n2 = 50013
x1, y1, z1, w1 = random_weighted_points(n1, Lbox, 0)
x2, y2, z2, w2 = random_weighted_points(n2, Lbox, 1)

start = time()
func(
    x1, y1, z1, w1, x2, y2, z2, w2, DEFAULT_RBINS_SQUARED, result)
end = time()
runtime = end-start

print('func:', func_str)
print('time:', runtime)
