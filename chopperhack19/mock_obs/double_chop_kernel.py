"""
"""
from numba import cuda


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

    g = 0
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
                #cuda.atomic.add(result, k-1, wprod)
                g += wprod
                k = k-1
                if k <= 0:
                    break
    result[0] += g
