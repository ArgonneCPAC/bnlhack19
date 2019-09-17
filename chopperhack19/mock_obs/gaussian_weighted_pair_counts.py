"""
"""
from numba import jit


@jit
def count_weighted_pairs_3d(x1, y1, z1, w1, x2, y2, z2, w2, rbins, result):
    """Naively count Npairs(<r), the total number of pairs that are separated
    by a distance less than r, for each r in the input rbins.
    """
    n1 = x1.size
    n2 = x2.size
    nbins = rbins.size
    rbins_squared = rbins*rbins

    for i in range(n1):
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

            k = nbins-1
            while dsq <= rbins_squared[k]:
                result[k] += wprod
                k=k-1
                if k<0:
                    break

    return result
