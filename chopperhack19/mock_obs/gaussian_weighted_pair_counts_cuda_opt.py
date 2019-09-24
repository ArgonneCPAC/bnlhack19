import numba
from numba import cuda
import math

__all__ = ('count_weighted_pairs_3d_cuda_smem',)


@cuda.jit()
def count_weighted_pairs_3d_cuda_smem(
        x1, y1, z1, w1, x2, y2, z2, w2, _rbins_squared, result):
    start = cuda.grid(1)
    stride = cuda.gridsize(1)

    n1 = x1.shape[0]
    n2 = x2.shape[0]
    nbins = _rbins_squared.shape[0]-1
    dlogr = math.log(
        math.sqrt(_rbins_squared[1]) / math.sqrt(_rbins_squared[0]))
    logminr = math.log(_rbins_squared[0]) / 2

    smem = cuda.shared.array(2049, numba.float32)
    if cuda.threadIdx.x == 0:
        for i in range(2049):
            smem[i] = 0
    cuda.syncthreads()

    for i in range(start, n1, stride):
        for j in range(n2):
            dx = x1[i] - x2[j]
            dy = y1[i] - y2[j]
            dz = z1[i] - z2[j]
            wp = w1[i] * w2[j]
            dsq = cuda.fma(dx, dx, cuda.fma(dy, dy, dz * dz))

            k = int((math.log(dsq)/2 - logminr) / dlogr)
            if k >= 0 and k < nbins:
                cuda.atomic.add(smem, k, wp)

    cuda.syncthreads()
    if cuda.threadIdx.x == 0:
        for k in range(nbins):
            cuda.atomic.add(result, k, smem[k])
