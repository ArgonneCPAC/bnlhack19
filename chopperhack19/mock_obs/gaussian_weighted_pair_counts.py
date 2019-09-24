"""
"""
import numpy as np
from numba import cuda, njit
import joblib
import multiprocessing

__all__ = (
    'count_weighted_pairs_3d_cuda',
    'count_weighted_pairs_3d_cuda_mesh',
    'count_weighted_pairs_3d_cpu_corrfunc',
    'count_weighted_pairs_3d_cpu_mp',
    'count_weighted_pairs_3d_cpu')

@cuda.jit
def count_weighted_pairs_3d_cuda_mesh(
        x1, y1, z1, w1, x2, y2, z2, w2, rbins_squared, result,
        nx, ny, nz, cell_id_indices, cell_id2_indices):
    """Naive pair counting with mesh in cuda. Note x/y/z/w are
    the sorted array output by calculate_chained_mesh.
    nx, ny, nz = mesh dimensions
    """
    start = cuda.grid(1)
    stride = cuda.gridsize(1)
    nbins = rbins_squared.shape[0] - 1    
    numcells = nx[0]*ny[0]*nz[0]
    for icell1 in range(start, numcells, stride):
        ifirst1 = cell_id_indices[icell1]
        ilast1 = cell_id_indices[icell1+1]

        x_icell1 = x1[ifirst1:ilast1]
        y_icell1 = y1[ifirst1:ilast1]
        z_icell1 = z1[ifirst1:ilast1]
        w_icell1 = w1[ifirst1:ilast1]
        Ni = ilast1 - ifirst1
        if Ni > 0:
            ix1 = icell1 // (ny[0]*nz[0])
            iy1 = (icell1 - ix1*ny[0]*nz[0]) // nz[0]
            iz1 = icell1 - (ix1*ny[0]*nz[0]) - (iy1*nz[0])

            leftmost_ix2 = max(0, ix1-1)
            leftmost_iy2 = max(0, iy1-1)
            leftmost_iz2 = max(0, iz1-1)

            rightmost_ix2 = min(ix1+2, nx[0])
            rightmost_iy2 = min(iy1+2, ny[0])
            rightmost_iz2 = min(iz1+2, nz[0])

            for icell2_ix in range(leftmost_ix2, rightmost_ix2):
                for icell2_iy in range(leftmost_iy2, rightmost_iy2):
                    for icell2_iz in range(leftmost_iz2, rightmost_iz2):

                        icell2 = icell2_ix*(ny[0]*nz[0]) + icell2_iy*nz[0] + icell2_iz
                        ifirst2 = cell_id2_indices[icell2]
                        ilast2 = cell_id2_indices[icell2+1]

                        x_icell2 = x2[ifirst2:ilast2]
                        y_icell2 = y2[ifirst2:ilast2]
                        z_icell2 = z2[ifirst2:ilast2]
                        w_icell2 = w2[ifirst2:ilast2]

                        Nj = ilast2 - ifirst2
                        if Nj > 0:
                            for i in range(0, Ni):
                                x1tmp = x_icell1[i]
                                y1tmp = y_icell1[i]
                                z1tmp = z_icell1[i] 
                                w1tmp = w_icell1[i]
                                for j in range(0, Nj):
                                    #calculate the square distance
                                    dx = x1tmp - x_icell2[j]
                                    dy = y1tmp - y_icell2[j]
                                    dz = z1tmp - z_icell2[j]
                                    wprod = w1tmp*w_icell2[j]
                                    dsq = dx*dx + dy*dy + dz*dz
     
                                    k = nbins-1
                                    while dsq <= rbins_squared[k]:
                                        cuda.atomic.add(result, k-1, wprod)
                                        k=k-1
                                        if k<0: break

@cuda.jit
def count_weighted_pairs_3d_cuda(
        x1, y1, z1, w1, x2, y2, z2, w2, rbins_squared, result):
    """Naively count Npairs(<r), the total number of pairs that are separated
    by a distance less than r, for each r**2 in the input rbins_squared.
    """
    start = cuda.grid(1)
    stride = cuda.gridsize(1)

    n1 = x1.shape[0]
    n2 = x2.shape[0]
    nbins = rbins_squared.shape[0]

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

            k = nbins-1
            while dsq <= rbins_squared[k]:
                cuda.atomic.add(result, k-1, wprod)
                k = k-1
                if k <= 0:
                    break


def count_weighted_pairs_3d_cpu_corrfunc(
        x1, y1, z1, w1, x2, y2, z2, w2, rbins_squared, result):
    # requires Corrfunc
    from Corrfunc.theory.DD import DD
    import multiprocessing
    rbins = np.sqrt(rbins_squared)
    threads = multiprocessing.cpu_count()
    _result = DD(
        0, threads, rbins, x1, y1, z1,
        weights1=w1, weight_type='pair_product',
        X2=x2, Y2=y2, Z2=z2, weights2=w2,
        periodic=False)
    result[:] = np.cumsum(_result['weightavg'] * _result['npairs'])


def count_weighted_pairs_3d_cpu_mp(
        x1, y1, z1, w1, x2, y2, z2, w2, rbins_squared, result):

    n1 = x1.shape[0]
    n_per = n1 // multiprocessing.cpu_count()
    if n_per < 1:
        n_per = 1

    jobs = []
    for i in range(multiprocessing.cpu_count()):
        start = i * n_per
        end = start + n_per
        if i == multiprocessing.cpu_count()-1:
            end = n1
        _result = np.zeros_like(result)
        jobs.append(joblib.delayed(count_weighted_pairs_3d_cpu)(
            x1[start:end],
            y1[start:end],
            z1[start:end],
            w1[start:end], x2, y2, z2, w2, rbins_squared, _result
        ))

    with joblib.Parallel(
            n_jobs=multiprocessing.cpu_count(), backend='loky') as p:
        res = p(jobs)

    result[:] = np.sum(np.stack(res, axis=1), axis=1)


@njit()
def count_weighted_pairs_3d_cpu(
        x1, y1, z1, w1, x2, y2, z2, w2, rbins_squared, result):

    n1 = x1.shape[0]
    n2 = x2.shape[0]
    nbins = rbins_squared.shape[0]

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
                result[k-1] += wprod
                k = k-1
                if k <= 0:
                    break

    return result
