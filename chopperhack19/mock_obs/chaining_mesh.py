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
