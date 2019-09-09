"""
"""
import numpy as np
from ..gaussian_weighted_histogram import numpy_gw_hist, numba_gw_hist


def test1():
    nbins, ndata = 25, 550
    xmin, xmax = 8, 12

    bins = np.linspace(xmin, xmax, nbins)

    rng = np.random.RandomState(43)
    data = rng.uniform(xmin, xmax, ndata)
    scale = 0.1

    numpy_khist = numpy_gw_hist(data, bins, scale)
    numba_khist = np.zeros_like(numpy_khist)
    numba_gw_hist(data, bins, scale, numba_khist)
    assert np.allclose(numpy_khist, numba_khist, atol=0.001)
