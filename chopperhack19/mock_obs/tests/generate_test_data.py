"""
"""
import numpy as np


__all__ = ('random_weighted_points', )


DEFAULT_SEED = 43


def random_weighted_points(n, Lbox, seed=DEFAULT_SEED):
    """
    """
    rng = np.random.RandomState(seed)
    data = rng.uniform(0, 1, n*4)
    x, y, z, w = data[:n]*Lbox, data[n:2*n]*Lbox, data[2*n:3*n]*Lbox, data[3*n:]
    return x, y, z, w
