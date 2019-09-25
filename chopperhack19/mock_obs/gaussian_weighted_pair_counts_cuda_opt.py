import numba
from numba import cuda
import math

__all__ = (
    'count_weighted_pairs_3d_cuda_smem_noncuml',
    'count_weighted_pairs_3d_cuda_noncuml',
    'count_weighted_pairs_3d_cuda_transpose_noncuml',
    'count_weighted_pairs_3d_cuda_revchop_noncuml',
    'count_weighted_pairs_3d_cuda_noncuml')


@cuda.jit(fastmath=True)
def count_weighted_pairs_3d_cuda_noncuml(
        x1, y1, z1, w1, x2, y2, z2, w2, _rbins_squared, result):
    start = cuda.grid(1)
    stride = cuda.gridsize(1)

    n1 = x1.shape[0]
    n2 = x2.shape[0]
    nbins = _rbins_squared.shape[0]-1
    dlogr = math.log(_rbins_squared[1] / _rbins_squared[0]) / 2
    logminr = math.log(_rbins_squared[0]) / 2

    for i in range(start, n1, stride):
        for j in range(n2):
            dx = x1[i] - x2[j]
            dy = y1[i] - y2[j]
            dz = z1[i] - z2[j]
            dsq = cuda.fma(dx, dx, cuda.fma(dy, dy, dz * dz))

            k = int((math.log(dsq)/2 - logminr) / dlogr)
            if k >= 0 and k < nbins:
                cuda.atomic.add(result, k, w1[i] * w2[j])


@cuda.jit(fastmath=True)
def count_weighted_pairs_3d_cuda_smem_noncuml(
        x1, y1, z1, w1, x2, y2, z2, w2, _rbins_squared, result):
    start = cuda.grid(1)
    stride = cuda.gridsize(1)

    n1 = x1.shape[0]
    n2 = x2.shape[0]
    nbins = _rbins_squared.shape[0]-1
    dlogr = math.log(
        _rbins_squared[1] / _rbins_squared[0]) / 2
    logminr = math.log(_rbins_squared[0]) / 2

    smem = cuda.shared.array(128, numba.float32)
    if cuda.threadIdx.x == 0:
        for i in range(128):
            smem[i] = 0
    cuda.syncthreads()

    for i in range(start, n1, stride):
        for j in range(n2):
            dx = x1[i] - x2[j]
            dy = y1[i] - y2[j]
            dz = z1[i] - z2[j]
            dsq = cuda.fma(dx, dx, cuda.fma(dy, dy, dz * dz))

            k = int((math.log(dsq)/2 - logminr) / dlogr)
            if k >= 0 and k < nbins:
                cuda.atomic.add(smem, k, w1[i] * w2[j])

    cuda.syncthreads()
    if cuda.threadIdx.x == 0:
        for k in range(nbins):
            cuda.atomic.add(result, k, smem[k])


@cuda.jit(fastmath=True)
def count_weighted_pairs_3d_cuda_revchop_noncuml(
        x1, y1, z1, w1, x2, y2, z2, w2, _rbins_squared, result):
    start = cuda.grid(1)
    stride = cuda.gridsize(1)

    n1 = x1.shape[0]
    n2 = x2.shape[0]
    nbins = _rbins_squared.shape[0]-1
    dlogr = math.log(
        _rbins_squared[1] / _rbins_squared[0]) / 2
    logminr = math.log(_rbins_squared[0]) / 2

    smem = cuda.shared.array(512, numba.float32)
    smem[cuda.threadIdx.x] = 0

    for k in range(1):
        for i in range(start, n1, stride):
            for j in range(n2):
                dx = x1[i] - x2[j]
                dy = y1[i] - y2[j]
                dz = z1[i] - z2[j]
                dsq = cuda.fma(dx, dx, cuda.fma(dy, dy, dz * dz))

                _k = int((math.log(dsq)/2 - logminr) / dlogr)
                if k == _k:
                    smem[cuda.threadIdx.x] = smem[cuda.threadIdx.x] + (w1[i] * w2[j])

        # cuda.syncthreads()
        #
        # i = numba.int32(cuda.blockDim.x) // 2
        # while i > 0:
        #     if cuda.threadIdx.x < i:
        #         smem[cuda.threadIdx.x] += smem[cuda.threadIdx.x + i]
        #     cuda.syncthreads()
        #     i = i // 2
        #
        # if cuda.threadIdx.x == 0:
        #     cuda.atomic.add(result, k, smem[0])
        #
        # cuda.syncthreads()


@cuda.jit(fastmath=True)
def count_weighted_pairs_3d_cuda_transpose_noncuml(
        ptswts1, ptswts2, _rbins_squared, result):
    start = cuda.grid(1)
    stride = cuda.gridsize(1)

    n1 = ptswts1.shape[0] // 4
    n2 = ptswts1.shape[0] // 4
    nbins = _rbins_squared.shape[0]-1
    dlogr = math.log(
        _rbins_squared[1] / _rbins_squared[0]) / 2
    logminr = math.log(_rbins_squared[0]) / 2

    smem = cuda.shared.array(128, numba.float32)
    if cuda.threadIdx.x == 0:
        for i in range(128):
            smem[i] = 0
    cuda.syncthreads()

    for i in range(start, n1, stride):
        loci = 4*i
        for j in range(n2):
            locj = 4*j
            dx = ptswts1[loci] - ptswts2[locj]
            dy = ptswts1[loci + 1] - ptswts2[locj + 1]
            dz = ptswts1[loci + 2] - ptswts2[locj + 2]
            dsq = cuda.fma(dx, dx, cuda.fma(dy, dy, dz * dz))

            k = int((math.log(dsq)/2 - logminr) / dlogr)
            if k >= 0 and k < nbins:
                cuda.atomic.add(smem, k, ptswts1[loci + 3] * ptswts2[locj + 3])

    cuda.syncthreads()
    if cuda.threadIdx.x == 0:
        for k in range(nbins):
            cuda.atomic.add(result, k, smem[k])
