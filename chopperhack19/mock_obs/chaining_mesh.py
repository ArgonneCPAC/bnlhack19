"""
"""
import numpy as np


def calculate_chaining_mesh(x, y, z, w, subvol_xlength, subvol_ylength, subvol_zlength,
            ndivs_x, ndivs_y, ndivs_z):
    """
    Parameters
    ----------
    x, y, z, w : ndarrays of shape (npts, )

    subvol_xyz_lengths : floats

    ndivs_xyz : ints

    Returns
    -------
    xout, yout, zout, wout: ndarrays of shape (npts, )
        Same as input points, sorted according to cell_id

    ixout, iyout, izout : integer ndarrays of shape (npts, )
        Digitized position of each point in each dimension, sorted according to cell_id

    cell_id_out : integer ndarray of shape (npts, )
        Unique integer specifying the cell defined by dictionary ordering of (ix, iy, iz)

    idx_sorted : integer ndarray of shape(npts, )
        Sorting array providing correspondence between input and output points:
        xout = x[idx_sorted]
        yout = y[idx_sorted]
        zout = z[idx_sorted]

    cell_id_indices : integer ndarray of shape (ndivs_x * ndivs_y * ndivs_z, )
    """
    ndivs_tot = ndivs_x * ndivs_y * ndivs_z
    npts = x.shape[0]

    xcell_size = subvol_xlength / ndivs_x
    ycell_size = subvol_ylength / ndivs_y
    zcell_size = subvol_zlength / ndivs_z

    ix = np.array(x // xcell_size, dtype='i4')
    iy = np.array(y // ycell_size, dtype='i4')
    iz = np.array(z // zcell_size, dtype='i4')

    cell_id = np.ravel_multi_index( (ix, iy, iz), (ndivs_x, ndivs_y, ndivs_z) )
    idx_sorted = np.ascontiguousarray(np.argsort(cell_id))
    cell_id_out = np.ascontiguousarray(cell_id[idx_sorted], dtype='i4')

    cell_id_indices = np.searchsorted(cell_id_out, range(ndivs_tot))
    cell_id_indices = np.ascontiguousarray(np.append(cell_id_indices, npts))

    xout = np.ascontiguousarray(x[idx_sorted], dtype='f4')
    yout = np.ascontiguousarray(y[idx_sorted], dtype='f4')
    zout = np.ascontiguousarray(z[idx_sorted], dtype='f4')
    wout = np.ascontiguousarray(w[idx_sorted], dtype='f4')

    ixout = np.ascontiguousarray(ix[idx_sorted], dtype='i4')
    iyout = np.ascontiguousarray(iy[idx_sorted], dtype='i4')
    izout = np.ascontiguousarray(iz[idx_sorted], dtype='i4')

    return xout, yout, zout, wout, ixout, iyout, izout, cell_id_out, idx_sorted, cell_id_indices


def get_double_chopped_sample2(x1, y1, z1, w1, x2, y2, z2, w2, nx1, ny1, nz1, nx2, ny2, nz2,
            rmax_x, rmax_y, rmax_z, xperiod, yperiod, zperiod):
    """
    Parameters
    ----------
    x1, y1, z1, w1 : ndarrays
        Float arrays of shape (npts1, ) storing xyz positions and weights of points in sample 1

    x2, y2, z2, w2 : ndarrays
        Float arrays of shape (npts2, ) storing xyz positions and weights of points in sample 2

    nx1, ny1, nz1 : integers
        Number of divisions in each dimension for sample 1

    nx2, ny2, nz2 : ints
        Number of divisions in each dimension for sample 1

    rmax_x, rmax_y, rmax_z : floats
        Maximum search length in each dimension

    xperiod, yperiod, zperiod : floats
        Size of the box containing the points in each dimension

    Returns
    -------
    x1out, y1out, z1out, w1out : ndarrays
        Float arrays of shape (npts1, ) storing xyz positions and weights of points in sample 1
        after sorting by chaining mesh ID

    cell1_indices : integer ndarray of shape (nx1 * ny1 * nz1, ) storing the indices
        of new cells in sample1

    x2out, y2out, z2out, w2out : ndarrays
        Float arrays of shape (n2out, ) storing xyz positions and weights of points in sample 2
        after copying the data required by the double-chop, so that n2out > npts2

    indx2 : list of length nx1*ny1*nz1
        Each element is a 2-element tuple storing (ifirst_cell1, ilast_cell1) such that
        x2out[ifirst_cell1:ilast_cell1] stores all points in the x-dimension
        that are located within the +/- rmax_x of the sample 1 points
        in the corresponding cell, and likewise for y2out, z2out, w2out
    """
    mesh1 = calculate_chaining_mesh(x1, y1, z1, w1, xperiod, yperiod, zperiod, nx1, ny1, nz1)
    mesh2 = calculate_chaining_mesh(x2, y2, z2, w2, xperiod, yperiod, zperiod, nx2, ny2, nz2)

    x1out, y1out, z1out, w1out, __, __, __, cell1_ids, __, cell1_indices = mesh1
    x2, y2, z2, w2, __, __, __, cell2_ids, __, __ = mesh2

    dx1 = xperiod / nx1
    dy1 = yperiod / ny1
    dz1 = zperiod / nz1

    dx2 = xperiod / nx2
    dy2 = yperiod / ny2
    dz2 = zperiod / nz2

    result = _get_double_chopped_sample2(nx1, ny1, nz1, dx1, dy1, dz1, rmax_x, rmax_y, rmax_z,
            dx2, dy2, dz2, nx2, ny2, nz2, x2, y2, z2, w2, cell2_ids)

    x2out, y2out, z2out, w2out, indx2 = result

    return x1out, y1out, z1out, w1out, cell1_indices, x2out, y2out, z2out, w2out, indx2


def _low_index_cell2(s1_low, rmax, ds2):
    return max(int((s1_low - rmax)/ds2), 0)


def _high_index_cell2(s1_high, rmax, ds2, n2):
    return min(int((s1_high + rmax)/ds2) + 1, n2-1)


def cell2_index_bounds(ix1, iy1, iz1, dx1, dy1, dz1,
                       rmax_x, rmax_y, rmax_z,
                       dx2, dy2, dz2, nx2, ny2, nz2):
    x1_low = ix1*dx1
    x1_high = (ix1+1)*dx1
    y1_low = iy1*dy1
    y1_high = (iy1+1)*dy1
    z1_low = iz1*dy1
    z1_high = (iz1+1)*dz1

    ix2_low = _low_index_cell2(x1_low, rmax_x, dx2)
    iy2_low = _low_index_cell2(y1_low, rmax_y, dy2)
    iz2_low = _low_index_cell2(z1_low, rmax_z, dz2)
    ix2_high = _high_index_cell2(x1_high, rmax_x, dx2, nx2)
    iy2_high = _high_index_cell2(y1_high, rmax_y, dy2, ny2)
    iz2_high = _high_index_cell2(z1_high, rmax_z, dz2, nz2)

    return ix2_low, ix2_high, iy2_low, iy2_high, iz2_low, iz2_high


def _generate_cell2_ids(ix1, iy1, iz1, dx1, dy1, dz1, rmax_x, rmax_y, rmax_z,
                  dx2, dy2, dz2, nx2, ny2, nz2):
    result = cell2_index_bounds(
        ix1, iy1, iz1, dx1, dy1, dz1, rmax_x, rmax_y, rmax_z,
        dx2, dy2, dz2, nx2, ny2, nz2)
    ix2_low, ix2_high, iy2_low, iy2_high, iz2_low, iz2_high = result
    for ix2 in range(ix2_low, ix2_high+1):
        assert ix2 < nx2, (ix2, nx2)
        for iy2 in range(iy2_low, iy2_high+1):
            assert iy2 < ny2, (iy2, ny2)
            for iz2 in range(iz2_low, iz2_high+1):
                assert iz2 < nz2, (iz2, nz2)
                try:
                    icell2 = np.ravel_multi_index( (ix2, iy2, iz2), (nx2, ny2, nz2) )
                except ValueError:
                    print((ix2, iy2, iz2), (nx2, ny2, nz2))
                yield icell2


def _calculate_sample2_mask(ix1, iy1, iz1, dx1, dy1, dz1, rmax_x, rmax_y, rmax_z,
                  dx2, dy2, dz2, nx2, ny2, nz2, cell2_ids):
    gen = _generate_cell2_ids(ix1, iy1, iz1, dx1, dy1, dz1, rmax_x, rmax_y, rmax_z,
                  dx2, dy2, dz2, nx2, ny2, nz2)
    mask = np.zeros_like(cell2_ids).astype(bool)
    for icell2 in gen:
        mask |= cell2_ids == icell2
    return mask


def _get_double_chopped_sample2(nx1, ny1, nz1, dx1, dy1, dz1, rmax_x, rmax_y, rmax_z,
            dx2, dy2, dz2, nx2, ny2, nz2, x2, y2, z2, w2, cell2_ids):
    """
    """
    ncells1 = nx1*ny1*nz1

    x2_collector = []
    y2_collector = []
    z2_collector = []
    w2_collector = []
    indx_collector = []

    ifirst = 0
    for icell1 in range(ncells1):
        ix1, iy1, iz1 = np.unravel_index(icell1, (nx1, ny1, nz1) )

        sample2_mask = _calculate_sample2_mask(
            ix1, iy1, iz1, dx1, dy1, dz1, rmax_x, rmax_y, rmax_z,
            dx2, dy2, dz2, nx2, ny2, nz2, cell2_ids)

        x2_collector.append(x2[sample2_mask])
        y2_collector.append(y2[sample2_mask])
        z2_collector.append(z2[sample2_mask])
        w2_collector.append(w2[sample2_mask])

        ilast = ifirst + np.count_nonzero(sample2_mask)
        indx_collector.append((ifirst, ilast))
        ifirst = ilast

    x2out = np.concatenate(x2_collector)
    y2out = np.concatenate(y2_collector)
    z2out = np.concatenate(z2_collector)
    w2out = np.concatenate(w2_collector)

    return x2out, y2out, z2out, w2out, indx_collector
