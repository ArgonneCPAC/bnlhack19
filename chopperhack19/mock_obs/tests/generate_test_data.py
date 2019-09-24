"""
"""
import numpy as np


__all__ = ('random_weighted_points', )


DEFAULT_SEED = 43
DEFAULT_NBINS = 20
DEFAULT_RMIN, DEFAULT_RMAX = 0.1, 40
DEFAULT_RBINS = np.logspace(
    np.log10(DEFAULT_RMIN), np.log10(DEFAULT_RMAX), DEFAULT_NBINS).astype(
        np.float32)
DEFAULT_RBINS_SQUARED = (DEFAULT_RBINS**2).astype(np.float32)


def random_weighted_points(n, Lbox, seed=DEFAULT_SEED):
    """
    """
    rng = np.random.RandomState(seed)
    data = rng.uniform(0, 1, n*4)
    x, y, z, w = (
        data[:n]*Lbox, data[n:2*n]*Lbox, data[2*n:3*n]*Lbox, data[3*n:])
    return (
        x.astype(np.float32), y.astype(np.float32), z.astype(np.float32),
        w.astype(np.float32))
