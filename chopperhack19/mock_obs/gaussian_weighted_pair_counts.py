"""
"""
import numpy as np
from numba import cuda, njit
import joblib
import multiprocessing

__all__ = (
    'count_weighted_pairs_3d_cuda',
    'count_weighted_pairs_3d_cpu_corrfunc',
    'count_weighted_pairs_3d_cpu_mp',
    'count_weighted_pairs_3d_cpu')


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
        X2=x2, Y2=y2, Z2=z2, weights2=w2)
    result[:] = np.cumsum(_result['weightavg']) * np.cumsum(_result['npairs'])


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
