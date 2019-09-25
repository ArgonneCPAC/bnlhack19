"""
"""
from numba import cuda, int32, float32


@cuda.jit
def double_chop_pairs_cuda(
        x1, y1, z1, w1, cell1, x2, y2, z2, w2, indx2, rbins_squared, result):
    """Naively count Npairs(<r), the total number of pairs that are separated
    by a distance less than r, for each r**2 in the input rbins_squared.
    """
    start = cuda.grid(1)
    stride = cuda.gridsize(1)

    n1 = x1.shape[0]
    nbins = rbins_squared.shape[0]

    for i in range(start, n1, stride):
        px = x1[i]
        py = y1[i]
        pz = z1[i]
        pw = w1[i]

        cell1_i = cell1[i]
        first = indx2[cell1_i]
        last = indx2[cell1_i+1]

        for j in range(first, last):
            qx = x2[j]
            qy = y2[j]
            qz = z2[j]
            qw = w2[j]

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


@cuda.jit(fastmath=True)
def double_chop_pairs_cuda_shmem_transpose(
        pt1, cell1, pt2, indx2, rbins_squared, result):
    """Naively count Npairs(<r), the total number of pairs that are separated
    by a distance less than r, for each r**2 in the input rbins_squared.
    """
    start = cuda.grid(1)
    stride = cuda.gridsize(1)

    n1 = pt1.shape[0]
    nbins = rbins_squared.shape[0]

    chunkSIZE = 512  # hard-coded to be the same as block size
    local_buffer = cuda.shared.array((chunkSIZE, 4), float32)

    for i in range(start, n1, stride):
        px, py, pz, pw = pt1[i]

        cell1_i = cell1[i]
        first = indx2[cell1_i]
        last = indx2[cell1_i+1]

        total = last - first
        n_chunks = (total + chunkSIZE - 1) // chunkSIZE

        # The first n_chunks-1 chuncks have size chunkSIZE
        for i_chunk in range(n_chunks-1):
            j = i_chunk * chunkSIZE + cuda.threadIdx.x
            if cuda.threadIdx.x < chunkSIZE:
                local_buffer[cuda.threadIdx.x, 0] = pt2[j, 0]
                local_buffer[cuda.threadIdx.x, 1] = pt2[j, 1]
                local_buffer[cuda.threadIdx.x, 2] = pt2[j, 2]
                local_buffer[cuda.threadIdx.x, 3] = pt2[j, 3]
            cuda.syncthreads()

            for q in range(chunkSIZE):
                qx, qy, qz, qw = local_buffer[q]

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

        # last chunk has fewer points
        j = (n_chunks-1) * chunkSIZE + cuda.threadIdx.x
        if j < total:
            local_buffer[cuda.threadIdx.x, 0] = pt2[j, 0]
            local_buffer[cuda.threadIdx.x, 1] = pt2[j, 1]
            local_buffer[cuda.threadIdx.x, 2] = pt2[j, 2]
            local_buffer[cuda.threadIdx.x, 3] = pt2[j, 3]
        cuda.syncthreads()

        last_chunk = total - (n_chunks-1) * chunkSIZE
        for q in range(last_chunk):
            qx, qy, qz, qw = local_buffer[q]

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
