import numpy as np
import numba
from numba import cuda
import math

__all__ = ('count_weighted_pairs_3d_cuda_smem',)


def count_weighted_pairs_3d_cuda_smem(
        x1, y1, z1, w1, x2, y2, z2, w2, rbins_squared, result):
    res = _count_weighted_pairs_3d_cuda_smem(
            x1, y1, z1, w1, x2, y2, z2, w2, rbins_squared, result)
    return np.cumsum(res)


@cuda.jit
def _count_weighted_pairs_3d_cuda_smem(
        x1, y1, z1, w1, x2, y2, z2, w2, rbins_squared, result):
    start = cuda.grid(1)
    stride = cuda.gridsize(1)

    n1 = x1.shape[0]
    n2 = x2.shape[0]
    nbins = rbins_squared.shape[0]-1
    logminr = math.log(rbins_squared[0])/2
    dlogr = math.log(rbins_squared[1])/2 - logminr

    lmem = cuda.local.array(1024, numba.float32)
    for i in range(1024):
        lmem[i] = 0

    smem = cuda.shared.array(1024, numba.float32)
    if cuda.threadIdx.x == 0:
        for i in range(1024):
            smem[i] = 0
    cuda.syncthreads()

    for i in range(start, n1, stride):
        px = x1[i]
        py = y1[i]
        pz = z1[i]
        pw = w1[i]
        for j in range(n2):
            qx = x2[j]
            qy = y2[j]
            qz = z2[j]
            qw = w2[j]
            dx = px-qx
            dy = py-qy
            dz = pz-qz
            wprod = pw*qw
            k = int(((math.log(dx*dx + dy*dy + dz*dz)/2) - logminr)/dlogr)

            if k >= 0 and k < nbins:
                lmem[k] += wprod

    for k in range(nbins):
        cuda.atomic.add(smem, k, lmem[k])
    cuda.syncthreads()
    if cuda.threadIdx.x == 0:
        for k in range(nbins):
            cuda.atomic.add(result, k, lmem[k])
