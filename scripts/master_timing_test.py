import numpy as np
import click

import chopperhack19.mock_obs
from chopperhack19.mock_obs.tests import random_weighted_points
from chopperhack19.mock_obs.tests.generate_test_data import (
    DEFAULT_RBINS_SQUARED)
from time import time


@click.command()
@click.option('--func', default='count_weighted_pairs_3d_cpu',
              help='the function to run')
@click.option('--blocks', default=512)
@click.option('--threads', default=512)
@click.option('--npoints', default=200013)
@click.option('--nmesh1', default=4)
@click.option('--nmesh2', default=16)
@click.option('--skip-numba-comp', is_flag=True, default=False)
def _main(func, blocks, threads, npoints, nmesh1, nmesh2, skip_numba_comp):

    func_str = func
    func = getattr(chopperhack19.mock_obs, func)
    if func is None:
        raise ImportError('could not import %s' % func_str)

    print('\nfunc_str:', func_str)
    print('func:', func)
    print('npoints:', npoints)
    if 'cuda' in func_str:
        from numba import cuda

        print('blocks:', blocks)
        print('threads:', threads)

    Lbox = 1000.
    result = np.zeros_like(DEFAULT_RBINS_SQUARED)[:-1]
    result = result.astype(np.float32)

    # get numba to compile once
    if not skip_numba_comp:
        n1 = 128
        n2 = 128
        _x1, _y1, _z1, _w1 = random_weighted_points(n1, Lbox, 0)
        _x2, _y2, _z2, _w2 = random_weighted_points(n2, Lbox, 1)
        if 'cuda_mesh' in func_str:
            print('nmesh1:', nmesh1)
            _cell_id_indices = np.zeros(len(_x1))
            _cell_id2_indices = np.zeros(len(_x1))
            _ndiv = np.array([nmesh1]*3, dtype=np.int32)
            _num_cell2_steps = np.array([1]*3, dtype=np.int32)
            func[blocks, threads](
                _x1, _y1, _z1, _w1, _x2, _y2, _z2, _w2,
                DEFAULT_RBINS_SQUARED, result,
                _ndiv, _cell_id_indices, _cell_id2_indices,
                _num_cell2_steps)
        elif ('double_chop' in func_str and
                'cuda' in func_str and 'transpose' in func_str):
            from chopperhack19.mock_obs import chaining_mesh as cm
            nx1 = nmesh1
            ny1 = nmesh1
            nz1 = nmesh1
            nx2 = nmesh2
            ny2 = nmesh2
            nz2 = nmesh2
            rmax_x = np.sqrt(DEFAULT_RBINS_SQUARED[-1])
            rmax_y = rmax_x
            rmax_z = rmax_y
            xperiod = Lbox
            yperiod = Lbox
            zperiod = Lbox
            (x1out, y1out, z1out, w1out, cell1out,
             x2out, y2out, z2out, w2out, indx2) = (
                cm.get_double_chopped_data(
                    _x1, _y1, _z1, _w1, _x2, _y2, _z2, _w2, nx1, ny1, nz1,
                    nx2, ny2, nz2,
                    rmax_x, rmax_y, rmax_z, xperiod, yperiod, zperiod))

            ptswts1 = np.empty((x1out.size, 4), dtype=np.float32)
            ptswts1[:, 0] = x1out
            ptswts1[:, 1] = y1out
            ptswts1[:, 2] = z1out
            ptswts1[:, 3] = w1out
            ptswts2 = np.empty((x2out.size, 4), dtype=np.float32)
            ptswts2[:, 0] = x2out
            ptswts2[:, 1] = y2out
            ptswts2[:, 2] = z2out
            ptswts2[:, 3] = w2out

            func[blocks, threads](
                ptswts1, cell1out, ptswts2, indx2, DEFAULT_RBINS_SQUARED, result)
        elif 'double_chop' in func_str:
            from chopperhack19.mock_obs import chaining_mesh as cm
            nx1 = nmesh1
            ny1 = nmesh1
            nz1 = nmesh1
            nx2 = nmesh2
            ny2 = nmesh2
            nz2 = nmesh2
            rmax_x = np.sqrt(DEFAULT_RBINS_SQUARED[-1])
            rmax_y = rmax_x
            rmax_z = rmax_y
            xperiod = Lbox
            yperiod = Lbox
            zperiod = Lbox
            (x1out, y1out, z1out, w1out, cell1out,
             x2out, y2out, z2out, w2out, indx2) = (
                cm.get_double_chopped_data(
                    _x1, _y1, _z1, _w1, _x2, _y2, _z2, _w2, nx1, ny1,
                    nz1, nx2, ny2, nz2,
                    rmax_x, rmax_y, rmax_z, xperiod, yperiod, zperiod))
            func[blocks, threads](
                x1out, y1out, z1out, w1out, cell1out,
                x2out, y2out, z2out, w2out, indx2,
                DEFAULT_RBINS_SQUARED, result)
        elif 'cuda_transpose2d' in func_str:
            ptswts1 = np.empty((_x1.size, 4), dtype=np.float32)
            ptswts1[:, 0] = _x1
            ptswts1[:, 1] = _y1
            ptswts1[:, 2] = _z1
            ptswts1[:, 3] = _w1
            ptswts2 = np.empty((_x2.size, 4), dtype=np.float32)
            ptswts2[:, 0] = _x2
            ptswts2[:, 1] = _y2
            ptswts2[:, 2] = _z2
            ptswts2[:, 3] = _w2

            func[(blocks, blocks), 1024](
                ptswts1, ptswts2, DEFAULT_RBINS_SQUARED, result)
        elif 'cuda_transpose' in func_str:
            ptswts1 = np.stack(
                [_x1, _y1, _z1, _w1], axis=1).ravel().astype(np.float32)
            ptswts2 = np.stack(
                [_x2, _y2, _z2, _w2], axis=1).ravel().astype(np.float32)

            func[blocks, threads](
                ptswts1, ptswts2, DEFAULT_RBINS_SQUARED, result)
        elif 'cuda' in func_str:
            func[blocks, threads](
                _x1, _y1, _z1, _w1, _x2, _y2, _z2, _w2,
                DEFAULT_RBINS_SQUARED, result)
        else:
            func(
                _x1, _y1, _z1, _w1, _x2, _y2, _z2, _w2,
                DEFAULT_RBINS_SQUARED, result)

    n1 = npoints
    n2 = npoints
    x1, y1, z1, w1 = random_weighted_points(n1, Lbox, 0)
    x2, y2, z2, w2 = random_weighted_points(n2, Lbox, 1)
    if 'cuda_mesh' in func_str:
        from chopperhack19.mock_obs import chaining_mesh as cm
        nx = nmesh1
        ny = nmesh1
        nz = nmesh1
        results = cm.calculate_chaining_mesh(
            x1, y1, z1, w1, Lbox, Lbox, Lbox, nx, ny, nz)
        (x1out, y1out, z1out, w1out, ixout, iyout, izout,
         cell_id_out, idx_sorted, cell_id_indices) = results
        results2 = cm.calculate_chaining_mesh(
            x2, y2, z2, w2, Lbox, Lbox, Lbox, nx, ny, nz)
        (x2out, y2out, z2out, w2out, ix2out, iy2out, iz2out,
         cell_id2_out, idx_sorted2, cell_id2_indices) = results2

        d_x1 = cuda.to_device(x1out.astype(np.float32))
        d_y1 = cuda.to_device(y1out.astype(np.float32))
        d_z1 = cuda.to_device(z1out.astype(np.float32))
        d_w1 = cuda.to_device(w1out.astype(np.float32))

        d_x2 = cuda.to_device(x2out.astype(np.float32))
        d_y2 = cuda.to_device(y2out.astype(np.float32))
        d_z2 = cuda.to_device(z2out.astype(np.float32))
        d_w2 = cuda.to_device(w2out.astype(np.float32))

        d_rbins_squared = cuda.to_device(
            DEFAULT_RBINS_SQUARED.astype(np.float32))
        d_result = cuda.device_array_like(result)
        d_ndiv = cuda.to_device(np.array([nmesh1]*3, dtype=np.int32))
        d_cell_id_indices = cuda.to_device(cell_id_indices)
        d_cell_id2_indices = cuda.to_device(cell_id2_indices)
        d_num_cell2_steps = cuda.to_device(np.array([1]*3, dtype=np.int32))
        start = time()
        for _ in range(3):
            func[blocks, threads](
                d_x1, d_y1, d_z1, d_w1, d_x2, d_y2, d_z2, d_w2,
                d_rbins_squared, d_result, d_ndiv,
                d_cell_id_indices, d_cell_id2_indices,
                d_num_cell2_steps)
            results_host = d_result.copy_to_host()
        end = time()
        assert np.all(np.isfinite(results_host))
        runtime = (end-start)/3
    elif ('double_chop' in func_str and
            'cuda' in func_str and 'transpose' in func_str):
        nx1 = nmesh1
        ny1 = nmesh1
        nz1 = nmesh1
        nx2 = nmesh2
        ny2 = nmesh2
        nz2 = nmesh2
        rmax_x = np.sqrt(DEFAULT_RBINS_SQUARED[-1])
        rmax_y = rmax_x
        rmax_z = rmax_y
        xperiod = Lbox
        yperiod = Lbox
        zperiod = Lbox
        (x1out, y1out, z1out, w1out, cell1out,
         x2out, y2out, z2out, w2out, indx2) = (
            cm.get_double_chopped_data(
                x1, y1, z1, w1, x2, y2, z2, w2, nx1, ny1, nz1, nx2, ny2, nz2,
                rmax_x, rmax_y, rmax_z, xperiod, yperiod, zperiod))

        ptswts1 = np.empty((x1out.size, 4), dtype=np.float32)
        ptswts1[:, 0] = x1out
        ptswts1[:, 1] = y1out
        ptswts1[:, 2] = z1out
        ptswts1[:, 3] = w1out
        ptswts2 = np.empty((x2out.size, 4), dtype=np.float32)
        ptswts2[:, 0] = x2out
        ptswts2[:, 1] = y2out
        ptswts2[:, 2] = z2out
        ptswts2[:, 3] = w2out

        d_ptswts1 = cuda.to_device(ptswts1)
        d_ptswts2 = cuda.to_device(ptswts2)
        d_cell1out = cuda.to_device(cell1out.astype(np.int32))
        d_indx2 = cuda.to_device(indx2.astype(np.int32))

        d_rbins_squared = cuda.to_device(
            DEFAULT_RBINS_SQUARED.astype(np.float32))
        d_result = cuda.device_array_like(result)

        start = time()
        for _ in range(3):
            func[blocks, threads](
                d_ptswts1, d_cell1out, d_ptswts2, d_indx2,
                d_rbins_squared, d_result)
            results_host = d_result.copy_to_host()
        end = time()
        assert np.all(np.isfinite(results_host))
        runtime = (end-start)/3

    elif 'double_chop' in func_str:
        nx1 = nmesh1
        ny1 = nmesh1
        nz1 = nmesh1
        nx2 = nmesh2
        ny2 = nmesh2
        nz2 = nmesh2
        rmax_x = np.sqrt(DEFAULT_RBINS_SQUARED[-1])
        rmax_y = rmax_x
        rmax_z = rmax_y
        xperiod = Lbox
        yperiod = Lbox
        zperiod = Lbox
        (x1out, y1out, z1out, w1out, cell1out,
         x2out, y2out, z2out, w2out, indx2) = (
            cm.get_double_chopped_data(
                x1, y1, z1, w1, x2, y2, z2, w2, nx1, ny1, nz1, nx2, ny2, nz2,
                rmax_x, rmax_y, rmax_z, xperiod, yperiod, zperiod))
        d_x1 = cuda.to_device(x1out.astype(np.float32))
        d_y1 = cuda.to_device(y1out.astype(np.float32))
        d_z1 = cuda.to_device(z1out.astype(np.float32))
        d_w1 = cuda.to_device(w1out.astype(np.float32))
        d_cell1out = cuda.to_device(cell1out.astype(np.int32))
        d_x2 = cuda.to_device(x2out.astype(np.float32))
        d_y2 = cuda.to_device(y2out.astype(np.float32))
        d_z2 = cuda.to_device(z2out.astype(np.float32))
        d_w2 = cuda.to_device(w2out.astype(np.float32))
        d_indx2 = cuda.to_device(indx2.astype(np.int32))
        d_rbins_squared = cuda.to_device(
            DEFAULT_RBINS_SQUARED.astype(np.float32))
        d_result = cuda.device_array_like(result)
        start = time()
        for _ in range(3):
            func[blocks, threads](
                d_x1, d_y1, d_z1, d_w1, d_cell1out,
                d_x2, d_y2, d_z2, d_w2, d_indx2,
                d_rbins_squared, d_result)
            results_host = d_result.copy_to_host()
        end = time()
        runtime = (end-start)/3
    elif 'cuda_transpose2d' in func_str:
        ptswts1 = np.empty((x1.size, 4), dtype=np.float32)
        ptswts1[:, 0] = x1
        ptswts1[:, 1] = y1
        ptswts1[:, 2] = z1
        ptswts1[:, 3] = w1
        ptswts2 = np.empty((x2.size, 4), dtype=np.float32)
        ptswts2[:, 0] = x2
        ptswts2[:, 1] = y2
        ptswts2[:, 2] = z2
        ptswts2[:, 3] = w2

        d_ptswts1 = cuda.to_device(ptswts1)
        d_ptswts2 = cuda.to_device(ptswts2)

        d_rbins_squared = cuda.to_device(
            DEFAULT_RBINS_SQUARED.astype(np.float32))
        d_result = cuda.device_array_like(result)

        start = time()
        for _ in range(3):
            func[(blocks, blocks), 1024](
                d_ptswts1, d_ptswts2, d_rbins_squared, d_result)
            results_host = d_result.copy_to_host()
        end = time()
        assert np.all(np.isfinite(results_host))
        runtime = (end-start)/3
    elif 'cuda_transpose' in func_str:
        ptswts1 = np.stack([x1, y1, z1, w1], axis=1).ravel().astype(np.float32)
        ptswts2 = np.stack([x2, y2, z2, w2], axis=1).ravel().astype(np.float32)

        d_ptswts1 = cuda.to_device(ptswts1)
        d_ptswts2 = cuda.to_device(ptswts2)

        d_rbins_squared = cuda.to_device(
            DEFAULT_RBINS_SQUARED.astype(np.float32))
        d_result = cuda.device_array_like(result)

        start = time()
        for _ in range(3):
            func[blocks, threads](
                d_ptswts1, d_ptswts2, d_rbins_squared, d_result)
            results_host = d_result.copy_to_host()
        end = time()
        assert np.all(np.isfinite(results_host))
        runtime = (end-start)/3
    elif 'cuda':
        d_x1 = cuda.to_device(x1.astype(np.float32))
        d_y1 = cuda.to_device(y1.astype(np.float32))
        d_z1 = cuda.to_device(z1.astype(np.float32))
        d_w1 = cuda.to_device(w1.astype(np.float32))

        d_x2 = cuda.to_device(x2.astype(np.float32))
        d_y2 = cuda.to_device(y2.astype(np.float32))
        d_z2 = cuda.to_device(z2.astype(np.float32))
        d_w2 = cuda.to_device(w2.astype(np.float32))

        d_rbins_squared = cuda.to_device(
            DEFAULT_RBINS_SQUARED.astype(np.float32))
        d_result = cuda.device_array_like(result)

        start = time()
        for _ in range(3):
            func[blocks, threads](
                d_x1, d_y1, d_z1, d_w1, d_x2, d_y2, d_z2, d_w2,
                d_rbins_squared, d_result)
            results_host = d_result.copy_to_host()
        end = time()
        assert np.all(np.isfinite(results_host))
        runtime = (end-start)/3
    else:
        d_x1 = x1
        d_y1 = y1
        d_z1 = z1
        d_w1 = w1

        d_x2 = x2
        d_y2 = y2
        d_z2 = z2
        d_w2 = w2

        d_rbins_squared = DEFAULT_RBINS_SQUARED
        d_result = result

        start = time()
        for _ in range(3):
            func(
                d_x1, d_y1, d_z1, d_w1, d_x2, d_y2, d_z2, d_w2,
                d_rbins_squared, d_result)
        end = time()
        runtime = (end-start)/3

    print('time:', runtime)
    print('result:', result)


if __name__ == '__main__':
    _main()
