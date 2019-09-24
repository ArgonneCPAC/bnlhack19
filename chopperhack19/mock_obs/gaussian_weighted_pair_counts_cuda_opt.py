import numba
from numba import cuda
import math

__all__ = ('count_weighted_pairs_3d_cuda_smem',)


@cuda.jit
def count_weighted_pairs_3d_cuda_smem(
        x1, y1, z1, w1, x2, y2, z2, w2, rbins_squared, result):
    start = cuda.grid(1)
    stride = cuda.gridsize(1)

    n1 = x1.shape[0]
    n2 = x2.shape[0]
    nbins = rbins_squared.shape[0]

    lmem = cuda.local.array(1024, numba.float32)
    # for i in range(1024):
    #     lmem[i] = 0
    #
    # smem = cuda.shared.array(1024, numba.float32)
    # if cuda.threadIdx.x == 0:
    #     for i in range(1024):
    #         smem[i] = 0
    # cuda.syncthreads()

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
            dsq = int(math.log(dx*dx + dy*dy + dz*dz)/2)

            lmem[1] += wprod
            # k = nbins-1
            # while dsq <= rbins_squared[k]:
            #     lmem[k-1] += wprod
            #     # cuda.atomic.add(smem, k-1, wprod)
            #     k = k-1
            #     if k <= 0:
            #         break

    # if cuda.threadIdx.x == 0:
    #     for i in range(1024):
    #         cuda.atomic.add(result, i, smem[i])

    # cuda.atomic.add(smem, 1, lmem[1])
    # cuda.syncthreads()
    # if cuda.threadIdx.x == 0:
    #     cuda.atomic.add(result, 1, lmem[1])
