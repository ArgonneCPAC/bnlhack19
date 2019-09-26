"""
"""
import numpy as np
from numpy import testing
from numba import cuda
import pytest

from .. import (
    count_weighted_pairs_3d_cpu,
    count_weighted_pairs_3d_cpu_mp,
    count_weighted_pairs_3d_cuda,
    count_weighted_pairs_3d_cpu_corrfunc,
    count_weighted_pairs_3d_cuda_transpose2d_smem,
    double_chop_pairs_cuda)
from .generate_test_data import random_weighted_points


DEFAULT_SEED = 43


@pytest.mark.parametrize('func', [
    count_weighted_pairs_3d_cpu_mp,
    count_weighted_pairs_3d_cpu_corrfunc])
def test_accuracy_cpu(func):
    # generate mocks
    n1 = 1000
    Lbox = 1000.
    x1, y1, z1, w1 = random_weighted_points(n1, Lbox, seed=DEFAULT_SEED)
    x2, y2, z2, w2 = random_weighted_points(n1, Lbox, seed=DEFAULT_SEED+1)

    # generate bins + result array
    nbins = 20
    rmin, rmax = 0.1, 40
    rbins = np.logspace(
        np.log10(rmin), np.log10(rmax), nbins).astype(np.float32)
    rbins_squared = rbins**2
    result_cpu = np.zeros(nbins-1)

    # run CPU test
    result_cpu_func = np.zeros(nbins-1)
    func(
        x1, y1, z1, w1, x2, y2, z2, w2, rbins_squared, result_cpu_func)

    # run CPU test
    count_weighted_pairs_3d_cpu(
        x1, y1, z1, w1, x2, y2, z2, w2, rbins_squared, result_cpu)

    # check if they are the same
    assert np.allclose(result_cpu_func, result_cpu, rtol=2e-7, atol=0)

@pytest.mark.parametrize('func', [
    count_weighted_pairs_3d_cuda_transpose2d_smem])
def test_accuracy_transpose(func):
    n1 = 1024
    Lbox = 1000.
    x1, y1, z1, w1 = random_weighted_points(n1, Lbox, seed=DEFAULT_SEED)
    x2, y2, z2, w2 = random_weighted_points(n1, Lbox, seed=DEFAULT_SEED+1)

    nbins = 20
    rmin, rmax = 0.1, 40
    rbins = np.logspace(
        np.log10(rmin), np.log10(rmax), nbins).astype(np.float32)
    rbins_squared = rbins**2
    result_cpu = np.zeros(nbins-1)

    ptswts1 = np.empty((x1.size, 4), dtype=np.float32)
    ptswts1[:, 0] = x1
    ptswts1[:, 1] = y1
    ptswts1[:, 2] = z1
    ptswts1[:, 3] = w1
    ptswts2 = np.empty((x2.size, 4), dtype=np.float32)
    ptswts2[:, 0] = x2
    ptswts2[:, 1] = y2
    ptswts2[:, 2] = z2
    ptswts2[:, 3] = w2

    d_ptswts1 = cuda.to_device(ptswts1)
    d_ptswts2 = cuda.to_device(ptswts2)
    d_rbins_squared = cuda.to_device(rbins_squared)
    d_result = cuda.to_device(result_cpu)

    count_weighted_pairs_3d_cpu(
        x1, y1, z1, w1, x2, y2, z2, w2, rbins_squared, result_cpu)
    func[(512,512),128](d_ptswts1, d_ptswts2, d_rbins_squared, d_result)
    result_gpu = d_result.copy_to_host()
    assert np.allclose(result_cpu, result_gpu, rtol=2e-7, atol=0)

@pytest.mark.parametrize('func', [
    double_chop_pairs_cuda])
def test_accuracy_double_chop(func):
    # generate mocks
    n1 = 1000
    Lbox = 1000.
    x1, y1, z1, w1 = random_weighted_points(n1, Lbox, seed=DEFAULT_SEED)
    x2, y2, z2, w2 = random_weighted_points(n1, Lbox, seed=DEFAULT_SEED+1)

    nbins = 20
    rmin, rmax = 0.1, 40
    rbins = np.logspace(
        np.log10(rmin), np.log10(rmax), nbins).astype(np.float32)
    rbins_squared = rbins**2
    result_cpu = np.zeros(nbins-1)

    # double chop the mocks
    from chopperhack19.mock_obs import chaining_mesh as cm
    nx = 4
    ny = 4
    nz = 4
    rmax_x = rbins[-1]
    rmax_y = rbins[-1]
    rmax_z = rbins[-1]
    (x1out, y1out, z1out, w1out, cell1out,
     x2out, y2out, z2out, w2out, indx2) = (
        cm.get_double_chopped_data(
            x1, y1, z1, w1, x2, y2, z2, w2, nx, ny, nz, nx, ny, nz,
            rmax_x, rmax_y, rmax_z, Lbox, Lbox, Lbox))
    d_x1 = cuda.to_device(x1out)
    d_y1 = cuda.to_device(y1out)
    d_z1 = cuda.to_device(z1out)
    d_w1 = cuda.to_device(w1out)
    d_cell1out = cuda.to_device(cell1out)
    d_x2 = cuda.to_device(x2out)
    d_y2 = cuda.to_device(y2out)
    d_z2 = cuda.to_device(z2out)
    d_w2 = cuda.to_device(w2out)
    d_indx2 = cuda.to_device(indx2)

    d_rbins_squared = cuda.to_device(rbins_squared)
    d_result = cuda.to_device(result_cpu)
    count_weighted_pairs_3d_cpu(
        x1, y1, z1, w1, x2, y2, z2, w2, rbins_squared, result_cpu)
    func(d_x1, d_y1, d_z1, d_w1, d_cell1out, d_x2, d_y2, d_z2, d_w2,
         d_indx2, d_rbins_squared, d_result)
    result_gpu = d_result.copy_to_host()
    assert np.allclose(result_cpu, result_gpu, rtol=2e-7, atol=0)


@pytest.mark.parametrize('func', [
    count_weighted_pairs_3d_cuda])
def test_accuracy_gpu(func):
    # generate mocks
    n1 = 1000
    Lbox = 1000.
    x1, y1, z1, w1 = random_weighted_points(n1, Lbox, seed=DEFAULT_SEED)
    x2, y2, z2, w2 = random_weighted_points(n1, Lbox, seed=DEFAULT_SEED+1)

    # generate bins + result array
    nbins = 20
    rmin, rmax = 0.1, 40
    rbins = np.logspace(
        np.log10(rmin), np.log10(rmax), nbins).astype(np.float32)
    rbins_squared = rbins**2
    result_cpu = np.zeros(nbins-1)

    # transfer over to device
    d_x1 = cuda.to_device(x1)
    d_y1 = cuda.to_device(y1)
    d_z1 = cuda.to_device(z1)
    d_w1 = cuda.to_device(w1)

    d_x2 = cuda.to_device(x2)
    d_y2 = cuda.to_device(y2)
    d_z2 = cuda.to_device(z2)
    d_w2 = cuda.to_device(w2)

    d_rbins_squared = cuda.to_device(rbins_squared)
    d_result_gpu = cuda.to_device(result_cpu)

    # run CPU test
    count_weighted_pairs_3d_cpu_corrfunc(
        x1, y1, z1, w1, x2, y2, z2, w2, rbins_squared, result_cpu)

    # run GPU test
    func(
        d_x1, d_y1, d_z1, d_w1, d_x2, d_y2, d_z2, d_w2,
        d_rbins_squared, d_result_gpu)
    # write back to host
    result_gpu = d_result_gpu.copy_to_host()

    # check if they are the same
    assert np.allclose(result_cpu, result_gpu, rtol=2e-7, atol=0)


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

    count_weighted_pairs_3d_cpu(
        pos1[:, 0], pos1[:, 1], pos1[:, 2], w1,
        pos2[:, 0], pos2[:, 1], pos2[:, 2], w2,
        rbins, result)

    assert ~np.all(result == 0)
