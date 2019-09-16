"""
"""
import numpy as np


from ..gaussian_weighted_pair_counts import count_weighted_pairs_3d


DEFAULT_SEED = 43


def test1():
    n1, n2 = 500, 1000

    rng1 = np.random.RandomState(DEFAULT_SEED)
    pos1 = rng1.uniform(0, 500, n1*3).reshape((n1, 3))
    w1 = rng1.uniform(0, 1, n1)

    rng2 = np.random.RandomState(DEFAULT_SEED)
    w2 = rng2.uniform(0, 1, n2)
    pos2 = rng2.uniform(0, 500, n2*3).reshape((n2, 3))

    rbins = np.logspace(-1, 1.5, 15)
    result = np.zeros_like(rbins)

    count_weighted_pairs_3d(
        pos1[:, 0], pos1[:, 1], pos1[:, 2], w1,
        pos2[:, 0], pos2[:, 1], pos2[:, 2], w2,
        rbins, result)

    assert ~np.all(result == 0)
