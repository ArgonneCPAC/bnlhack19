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
    'count_weighted_pairs_3d_cuda_transpose2d_smem',
    'count_weighted_pairs_3d_cuda_transpose2d',
    'count_weighted_pairs_3d_cuda_smem',
    'count_weighted_pairs_3d_cuda_extrabins',
    'count_weighted_pairs_3d_cuda_transpose2d_smem_blocksonly')


# @cuda.jit(fastmath=True)
# def count_weighted_pairs_3d_cuda_noncuml(
#         x1, y1, z1, w1, x2, y2, z2, w2, _rbins_squared, result):
#     start = cuda.grid(1)
#     stride = cuda.gridsize(1)
#
#     n1 = x1.shape[0]
#     n2 = x2.shape[0]
#     nbins = _rbins_squared.shape[0]-1
#     dlogr = math.log(_rbins_squared[1] / _rbins_squared[0]) / 2
#     logminr = math.log(_rbins_squared[0]) / 2
#
#     for i in range(start, n1, stride):
#         for j in range(n2):
#             dx = x1[i] - x2[j]
#             dy = y1[i] - y2[j]
#             dz = z1[i] - z2[j]
#             dsq = cuda.fma(dx, dx, cuda.fma(dy, dy, dz * dz))
#
#             k = int((math.log(dsq)/2 - logminr) / dlogr)
#             if k >= 0 and k < nbins:
#                 cuda.atomic.add(result, k, w1[i] * w2[j])


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


@cuda.jit(fastmath=True, max_registers=32)
def count_weighted_pairs_3d_cuda_transpose2d_smem(
        pt1, pt2, rbins_squared, result):
    """Naively count Npairs(<r), the total number of pairs that are separated
    by a distance less than r, for each r**2 in the input rbins_squared.
    """
    n1 = pt1.shape[0] // cuda.gridDim.x
    n2 = pt2.shape[0] // cuda.gridDim.y
    nbins = rbins_squared.shape[0]

    loc_1 = cuda.blockIdx.x * n1
    loc_2 = cuda.blockIdx.y * n2

    chunk_size = 32
    local_buffer1 = cuda.shared.array((chunk_size, 4), numba.float32)
    local_buffer2 = cuda.shared.array((chunk_size, 4), numba.float32)

    n_loads = (chunk_size + cuda.blockDim.x - 1) // cuda.blockDim.x
    n_comps = cuda.blockDim.x // chunk_size

    n_chunks1 = (n1 + chunk_size - 1) // chunk_size
    n_chunks2 = (n2 + chunk_size - 1) // chunk_size

    assert n_chunks1 * chunk_size * cuda.gridDim.x == pt1.shape[0]
    assert n_chunks2 * chunk_size * cuda.gridDim.y == pt2.shape[0]

    for chunk1 in range(n_chunks1):
        # do load
        for load in range(n_loads):
            tloc = n_loads * cuda.threadIdx.x + load
            if tloc < chunk_size:
                ploc = (
                    loc_1 +
                    chunk1 * chunk_size +
                    tloc)
                local_buffer1[tloc, 0] = pt1[ploc, 0]
                local_buffer1[tloc, 1] = pt1[ploc, 1]
                local_buffer1[tloc, 2] = pt1[ploc, 2]
                local_buffer1[tloc, 3] = pt1[ploc, 3]

        for chunk2 in range(n_chunks2):
            # do load
            for load in range(n_loads):
                tloc = n_loads * cuda.threadIdx.x + load
                if tloc < chunk_size:
                    ploc = (
                        loc_2 +
                        chunk2 * chunk_size +
                        tloc)
                    local_buffer2[tloc, 0] = pt2[ploc, 0]
                    local_buffer2[tloc, 1] = pt2[ploc, 1]
                    local_buffer2[tloc, 2] = pt2[ploc, 2]
                    local_buffer2[tloc, 3] = pt2[ploc, 3]
            cuda.syncthreads()

            # let the threads each handle one thing
            for load in range(n_loads):
                tloc = (n_loads * cuda.threadIdx.x + load) // n_comps
                if tloc < chunk_size:
                    px, py, pz, pw = local_buffer1[tloc]

                    comp_start = (
                        n_loads * cuda.threadIdx.x + load - tloc * n_comps)
                    for q in range(comp_start, chunk_size, n_comps):
                        qx, qy, qz, qw = local_buffer2[q]

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


@cuda.jit(fastmath=True, max_registers=32)
def count_weighted_pairs_3d_cuda_transpose2d(
        pt1, pt2, rbins_squared, result):
    """Naively count Npairs(<r), the total number of pairs that are separated
    by a distance less than r, for each r**2 in the input rbins_squared.
    """
    n1 = pt1.shape[0]
    n2 = pt2.shape[0]
    nbins = rbins_squared.shape[0]

    start = cuda.grid(1)
    stride = cuda.gridsize(1)

    for i in range(start, n1, stride):
        px, py, pz, pw = pt1[i]

        for j in range(n2):
            qx, qy, qz, qw = pt2[j]

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


@cuda.jit(fastmath=True)  # , max_registers=32)
def count_weighted_pairs_3d_cuda_smem(
        x1, y1, z1, w1, x2, y2, z2, w2, rbins_squared, result):
    """Naively count Npairs(<r), the total number of pairs that are separated
    by a distance less than r, for each r**2 in the input rbins_squared.
    """
    n1 = x1.shape[0]
    n2 = x2.shape[0]
    nbins = rbins_squared.shape[0]

    start = cuda.grid(1)
    stride = cuda.gridsize(1)

    chunk_size = 64
    buff = cuda.shared.array((chunk_size, 4), numba.float32)

    n_loads = (chunk_size + cuda.blockDim.x - 1) // cuda.blockDim.x

    n_chunks = (n2 + chunk_size - 1) // chunk_size

    for i in range(start, n1, stride):
        px = x1[i]
        py = y1[i]
        pz = z1[i]
        pw = w1[i]

        for chunk in range(n_chunks-1):
            for load in range(n_loads):
                tloc = n_loads * cuda.threadIdx.x + load
                if tloc < chunk_size:
                    ploc = chunk * chunk_size + tloc
                    buff[tloc, 0] = x2[ploc]
                    buff[tloc, 1] = y2[ploc]
                    buff[tloc, 2] = z2[ploc]
                    buff[tloc, 3] = w2[ploc]
            cuda.syncthreads()

            for q in range(chunk_size):
                qx, qy, qz, qw = buff[q]

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

        last_chunk = n2 - (n_chunks-1) * chunk_size
        for load in range(n_loads):
            tloc = n_loads * cuda.threadIdx.x + load
            if tloc < last_chunk:
                ploc = chunk * chunk_size + tloc
                buff[tloc, 0] = x2[ploc]
                buff[tloc, 1] = y2[ploc]
                buff[tloc, 2] = z2[ploc]
                buff[tloc, 3] = w2[ploc]
        cuda.syncthreads()

        for q in range(last_chunk):
            qx, qy, qz, qw = buff[q]

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


@cuda.jit
def count_weighted_pairs_3d_cuda_extrabins(
        x1, y1, z1, w1, x2, y2, z2, w2, rbins_squared, result):
    """Naively count Npairs(<r), the total number of pairs that are separated
    by a distance less than r, for each r**2 in the input rbins_squared.
    """
    start = cuda.grid(1)
    stride = cuda.gridsize(1)

    n1 = x1.shape[0]
    n2 = x2.shape[0]
    nbins_minus1 = rbins_squared.shape[0] - 1

    dlogr = math.log(rbins_squared[2]/rbins_squared[1])/2
    minlogr = math.log(rbins_squared[1])/2

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
            dsq = dx*dx + dy*dy + dz*dz

            k = int((math.log(dsq)/2 - minlogr) / dlogr)
            k = min(max(k, 0), nbins_minus1)
            cuda.atomic.add(result, k+1, wprod)


@cuda.jit
def count_weighted_pairs_3d_cuda_noncuml(
        x1, y1, z1, w1, x2, y2, z2, w2, rbins_squared, result):
    """Naively count Npairs(<r), the total number of pairs that are separated
    by a distance less than r, for each r**2 in the input rbins_squared.
    """
    start = cuda.grid(1)
    stride = cuda.gridsize(1)

    n1 = x1.shape[0]
    n2 = x2.shape[0]
    nbins = rbins_squared.shape[0] - 1

    dlogr = math.log(rbins_squared[1]/rbins_squared[0])/2
    minlogr = math.log(rbins_squared[0])/2

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
            dsq = dx*dx + dy*dy + dz*dz

            k = int((math.log(dsq)/2 - minlogr) / dlogr)
            if k >= 0 and k < nbins:
                cuda.atomic.add(result, k, wprod)


@cuda.jit(fastmath=True, max_registers=32)
def count_weighted_pairs_3d_cuda_transpose2d_smem_blocksonly(
        pt1, pt2, rbins_squared, result):
    """Naively count Npairs(<r), the total number of pairs that are separated
    by a distance less than r, for each r**2 in the input rbins_squared.
    """
    n1 = pt1.shape[0] // cuda.gridDim.x
    n2 = pt2.shape[0] // cuda.gridDim.y
    nbins = rbins_squared.shape[0]

    loc_1 = cuda.blockIdx.x * n1
    loc_2 = cuda.blockIdx.y * n2

    chunk_size = 128
    local_buffer1 = cuda.shared.array((chunk_size, 4), numba.float32)
    local_buffer2 = cuda.shared.array((chunk_size, 4), numba.float32)

    ploc = loc_1 + cuda.threadIdx.x
    local_buffer1[cuda.threadIdx.x, 0] = pt1[ploc, 0]
    local_buffer1[cuda.threadIdx.x, 1] = pt1[ploc, 1]
    local_buffer1[cuda.threadIdx.x, 2] = pt1[ploc, 2]
    local_buffer1[cuda.threadIdx.x, 3] = pt1[ploc, 3]

    ploc = loc_2 + cuda.threadIdx.x
    local_buffer2[cuda.threadIdx.x, 0] = pt2[ploc, 0]
    local_buffer2[cuda.threadIdx.x, 1] = pt2[ploc, 1]
    local_buffer2[cuda.threadIdx.x, 2] = pt2[ploc, 2]
    local_buffer2[cuda.threadIdx.x, 3] = pt2[ploc, 3]

    cuda.syncthreads()

    px, py, pz, pw = local_buffer1[cuda.threadIdx.x]

    for q in range(chunk_size):
        qx, qy, qz, qw = local_buffer2[q]

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
