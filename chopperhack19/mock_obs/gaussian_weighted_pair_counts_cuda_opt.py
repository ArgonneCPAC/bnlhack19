import numba
from numba import cuda
import math
# import jinja2

__all__ = (  # noqa
    'count_weighted_pairs_3d_cuda_smem_noncuml',
    'count_weighted_pairs_3d_cuda_noncuml',
    'count_weighted_pairs_3d_cuda_transpose_noncuml',
    'count_weighted_pairs_3d_cuda_smemload_noncuml',
    # 'count_weighted_pairs_3d_cuda_revchop_noncuml',
    'count_weighted_pairs_3d_cuda_noncuml_pairsonly',
    'count_weighted_pairs_3d_cuda_noncuml',
    'count_weighted_pairs_3d_cuda_smemload_noncuml_pairsonly',
    'count_weighted_pairs_3d_cuda_transpose2d_smem')


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
def count_weighted_pairs_3d_cuda_noncuml_pairsonly(
        x1, y1, z1, w1, x2, y2, z2, w2, _rbins_squared, result):
    start = cuda.grid(1)
    stride = cuda.gridsize(1)

    n1 = x1.shape[0]
    n2 = x2.shape[0]
    nbins = _rbins_squared.shape[0]-1
    dlogr = math.log(_rbins_squared[1] / _rbins_squared[0]) / 2
    logminr = math.log(_rbins_squared[0]) / 2

    g = 0
    for i in range(start, n1, stride):
        for j in range(n2):
            dx = x1[i] - x2[j]
            dy = y1[i] - y2[j]
            dz = z1[i] - z2[j]
            dsq = cuda.fma(dx, dx, cuda.fma(dy, dy, dz * dz))

            k = int((math.log(dsq)/2 - logminr) / dlogr)
            if k >= 0 and k < nbins:
                g += (w1[i] * w2[j])

    result[0] += g


SMEM_CHUNK_SIZE = 2048


@cuda.jit(fastmath=True)
def count_weighted_pairs_3d_cuda_smemload_noncuml(
        x1, y1, z1, w1, x2, y2, z2, w2, _rbins_squared, result):
    start = cuda.grid(1)
    stride = cuda.gridsize(1)

    n1 = x1.shape[0]
    n2 = x2.shape[0]
    nbins = _rbins_squared.shape[0]-1
    dlogr = math.log(
        _rbins_squared[1] / _rbins_squared[0]) / 2
    logminr = math.log(_rbins_squared[0]) / 2

    n_chunks = n2 // SMEM_CHUNK_SIZE
    if n_chunks * SMEM_CHUNK_SIZE < n2:
        n_chunks += 1

    sx = cuda.shared.array(SMEM_CHUNK_SIZE, numba.float32)
    sy = cuda.shared.array(SMEM_CHUNK_SIZE, numba.float32)
    sz = cuda.shared.array(SMEM_CHUNK_SIZE, numba.float32)
    sw = cuda.shared.array(SMEM_CHUNK_SIZE, numba.float32)

    n_loads = SMEM_CHUNK_SIZE // cuda.blockDim.x

    for i in range(start, n1, stride):
        for chunk in range(n_chunks):
            loc = chunk * SMEM_CHUNK_SIZE
            endloc = loc + SMEM_CHUNK_SIZE
            if endloc > n2:
                endloc = n2
            tmax = endloc - loc
            for l in range(n_loads):
                idx = n_loads * cuda.threadIdx.x + l
                if idx < tmax:
                    midx = loc + idx
                    sx[idx] = x2[midx]
                    sy[idx] = y2[midx]
                    sz[idx] = z2[midx]
                    sw[idx] = w2[midx]
            cuda.syncthreads()

            for j in range(tmax):
                dx = x1[i] - sx[j]
                dy = y1[i] - sy[j]
                dz = z1[i] - sz[j]
                dsq = cuda.fma(dx, dx, cuda.fma(dy, dy, dz * dz))

                k = int((math.log(dsq)/2 - logminr) / dlogr)
                if k >= 0 and k < nbins:
                    cuda.atomic.add(result, k, w1[i] * sw[j])


@cuda.jit(fastmath=True)
def count_weighted_pairs_3d_cuda_smemload_noncuml_pairsonly(
        x1, y1, z1, w1, x2, y2, z2, w2, _rbins_squared, result):
    start = cuda.grid(1)
    stride = cuda.gridsize(1)

    n1 = x1.shape[0]
    n2 = x2.shape[0]
    nbins = _rbins_squared.shape[0]-1
    dlogr = math.log(
        _rbins_squared[1] / _rbins_squared[0]) / 2
    logminr = math.log(_rbins_squared[0]) / 2

    n_chunks = n2 // SMEM_CHUNK_SIZE
    if n_chunks * SMEM_CHUNK_SIZE < n2:
        n_chunks += 1

    sx = cuda.shared.array(SMEM_CHUNK_SIZE, numba.float32)
    sy = cuda.shared.array(SMEM_CHUNK_SIZE, numba.float32)
    sz = cuda.shared.array(SMEM_CHUNK_SIZE, numba.float32)
    sw = cuda.shared.array(SMEM_CHUNK_SIZE, numba.float32)

    n_loads = SMEM_CHUNK_SIZE // cuda.blockDim.x

    g = 0
    for i in range(start, n1, stride):
        for chunk in range(n_chunks):
            loc = chunk * SMEM_CHUNK_SIZE
            endloc = loc + SMEM_CHUNK_SIZE
            if endloc > n2:
                endloc = n2
            tmax = endloc - loc
            for l in range(n_loads):
                idx = n_loads * cuda.threadIdx.x + l
                if idx < tmax:
                    midx = loc + idx
                    sx[idx] = x2[midx]
                    sy[idx] = y2[midx]
                    sz[idx] = z2[midx]
                    sw[idx] = w2[midx]
            cuda.syncthreads()

            for j in range(tmax):
                dx = x1[i] - sx[j]
                dy = y1[i] - sy[j]
                dz = z1[i] - sz[j]
                dsq = cuda.fma(dx, dx, cuda.fma(dy, dy, dz * dz))

                k = int((math.log(dsq)/2 - logminr) / dlogr)
                if k >= 0 and k < nbins:
                    g += (w1[i] * sw[j])

    result[0] += g


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


# exec(jinja2.Template("""
# @cuda.jit(fastmath=True)
# def count_weighted_pairs_3d_cuda_revchop_noncuml(
#        x1, y1, z1, w1, x2, y2, z2, w2, _rbins_squared, result):
#    start = cuda.grid(1)
#    stride = cuda.gridsize(1)
#
#    n1 = x1.shape[0]
#    n2 = x2.shape[0]
#    nbins = _rbins_squared.shape[0]-1
#    dlogr = math.log(
#        _rbins_squared[1] / _rbins_squared[0]) / 2
#    logminr = math.log(_rbins_squared[0]) / 2
#
#    smem = cuda.shared.array(512, numba.float32)
#
#    {% for bin in range({{ n_bins }}) %}
#    g{{ bin }} = 0
#    {% endfor %}
#    for i in range(start, n1, stride):
#        for j in range(n2):
#            dx = x1[i] - x2[j]
#            dy = y1[i] - y2[j]
#            dz = z1[i] - z2[j]
#            dsq = cuda.fma(dx, dx, cuda.fma(dy, dy, dz * dz))
#
#            k = int((math.log(dsq)/2 - logminr) / dlogr)
#            if k == 0:
#                g0 += (w1[i] * w2[j])
#            {% for bin in range(1, 16) %}
#            elif k == {{ bin }}:
#                g{{ bin }} += (w1[i] * w2[j])
#            {% endfor %}
#
#    for k in range({{ n_bins }}):
#        if k == 0:
#            smem[cuda.threadIdx.x] = g0
#        {% for bin in range(1, {{ n_bins }}) %}
#        elif k == {{ bin }}:
#            smem[cuda.threadIdx.x] = g{{ bin }}
#        {% endfor %}
#        cuda.syncthreads()
#
#        i = numba.int32(cuda.blockDim.x) // 2
#        while i > 32:
#            if cuda.threadIdx.x < i:
#                smem[cuda.threadIdx.x] += smem[cuda.threadIdx.x + i]
#            cuda.syncthreads()
#            i = i >> 1
#
#        if cuda.threadIdx.x < 32:
#            smem[cuda.threadIdx.x] += smem[cuda.threadIdx.x + 32]
#            smem[cuda.threadIdx.x] += smem[cuda.threadIdx.x + 16]
#            smem[cuda.threadIdx.x] += smem[cuda.threadIdx.x + 8]
#            smem[cuda.threadIdx.x] += smem[cuda.threadIdx.x + 4]
#            smem[cuda.threadIdx.x] += smem[cuda.threadIdx.x + 2]
#            smem[cuda.threadIdx.x] += smem[cuda.threadIdx.x + 1]
#
#        if cuda.threadIdx.x == 0:
#            cuda.atomic.add(result, k, smem[0])
# """).render(n_bins=2))


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


@cuda.jit(fastmath=True)
def count_weighted_pairs_3d_cuda_transpose2d_smem(
        pt1, pt2, rbins_squared, result):
    """Naively count Npairs(<r), the total number of pairs that are separated
    by a distance less than r, for each r**2 in the input rbins_squared.
    """
    #  pt1 runs along the x-direction
    #  pt2 runs along the y-direction
    #  cuda.gridDim.x = Number of blocks in x-dimension
    #  cuda.gridDim.y = Number of blocks in y-dimension
    n1 = pt1.shape[0] // cuda.gridDim.x  #  Number of pt1 points per block
    n2 = pt2.shape[0] // cuda.gridDim.y  #  Number of pt2 points per block
    nbins = rbins_squared.shape[0]

    loc_1 = cuda.blockIdx.x * n1  #  Index of the first point in the pt1 block
    loc_2 = cuda.blockIdx.y * n2  #  Index of the first point in the pt2 block

    chunk_size = 1024  #  Number of points in each chunk - should divide number of threads per block
    chunk1_buffer = cuda.shared.array((chunk_size, 4), numba.float32)  #  smem for pt1 in the chunk
    chunk2_buffer = cuda.shared.array((chunk_size, 4), numba.float32)  #  smem for pt2 in the chunk

    n_chunks1 = (n1 + chunk_size - 1) // chunk_size  #  Number of chunks needed to loop over all of pt1
    n_chunks2 = (n2 + chunk_size - 1) // chunk_size  #  Number of chunks needed to loop over all of pt2

    for chunk1 in range(n_chunks1):
        #  loc_1 is the index of the first data point in the block
        #  Within each block, chunk1 * chunk_size = index of the first data point in the chunk
        #  Within each chunk, threadIdx.x = index of the data point assigned to the thread
        #  So jump to the block, then to the chunk, then to the thread
        loc = loc_1 + chunk1 * chunk_size + cuda.threadIdx.x
        chunk1_buffer[cuda.threadIdx.x, 0] = pt1[loc, 0]
        chunk1_buffer[cuda.threadIdx.x, 1] = pt1[loc, 1]
        chunk1_buffer[cuda.threadIdx.x, 2] = pt1[loc, 2]
        chunk1_buffer[cuda.threadIdx.x, 3] = pt1[loc, 3]
        cuda.syncthreads()  #  Syncs all threads in the block (not actually necessary)

        for chunk2 in range(n_chunks2):
            #  Same gymnastics for the pt2 data
            loc = loc_2 + chunk2 * chunk_size + cuda.threadIdx.x
            chunk2_buffer[cuda.threadIdx.x, 0] = pt2[loc, 0]
            chunk2_buffer[cuda.threadIdx.x, 1] = pt2[loc, 1]
            chunk2_buffer[cuda.threadIdx.x, 2] = pt2[loc, 2]
            chunk2_buffer[cuda.threadIdx.x, 3] = pt2[loc, 3]
            cuda.syncthreads()  #  Syncs all threads in the block (critical)

            #  Grab the pt1 point that belongs to the thread
            px, py, pz, pw = chunk1_buffer[cuda.threadIdx.x]

            #  Loop over every pt2 point that belongs to the thread
            for q in range(chunk_size):
                qx, qy, qz, qw = chunk2_buffer[q]

                dx = px-qx
                dy = py-qy
                dz = pz-qz
                wprod = pw*qw
                dsq = dx*dx + dy*dy + dz*dz

                k = nbins-1
                while dsq <= rbins_squared[k]:
                    cuda.atomic.add(result, k-1, wprod)
                    k = k-1
                    if k <= 0:
                        break
