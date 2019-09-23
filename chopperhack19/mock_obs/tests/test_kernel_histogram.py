"""
"""
import numpy as np
from numpy import testing
from numba import cuda
from ..gaussian_weighted_histogram import numpy_gw_hist, numba_gw_hist, cuda_gw_hist

def test_gpu_accuracy():
    # generate mock data
    rng = np.random.RandomState(42)
    n1 = 10000
    Lbox = 1000.
    scale = 10 # numba_gw_hist doesn't like an array of scales
    scales = np.zeros(n1).astype(np.float64)+10
    x = rng.uniform(0, Lbox, size=n1).astype(np.float64)
    nbins = 20
    rmin, rmax = 0.1, 40
    rbins = np.logspace(np.log10(rmin), np.log10(rmax), nbins).astype(np.float64)
    result_cpu = np.zeros(nbins-1).astype(np.float64) 

    # transfer things over to device
    d_x = cuda.to_device(x)
    d_rbins = cuda.to_device(rbins)
    d_scales = cuda.to_device(scales)
    d_result_gpu = cuda.to_device(result_cpu)

    # test on CPU
    numba_gw_hist(x, rbins, scale, result_cpu)
    # test on GPU
    cuda_gw_hist(d_x, d_rbins, d_scales, d_result_gpu)
    result_gpu = d_result_gpu.copy_to_host()

    # for some reason this one takes an almost equal. I blame the erf. It agrees within reason.
    testing.assert_array_almost_equal(result_cpu, result_gpu)

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
