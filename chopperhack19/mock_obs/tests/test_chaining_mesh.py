"""
"""
import numpy as np
from chaining_mesh import calculate_chaining_mesh
from thechopper import points_in_buffered_rectangle

DEFAULT_SEED = 43


def random_points(n, Lbox, seed=DEFAULT_SEED):
    """
    """
    rng = np.random.RandomState(seed)
    data = rng.uniform(0, 1, n*4)
    x, y, z = data[:n]*Lbox, data[n:2*n]*Lbox, data[2*n:3*n]*Lbox
    return x, y, z


def test1():
    n = 1234
    Lbox = 1000.

    nx, ny, nz = 5, 5, 5

    x, y, z = random_points(n, Lbox)

    results = calculate_chaining_mesh(x, y, z, Lbox, Lbox, Lbox, nx, ny, nz)
    xout, yout, zout, ixout, iyout, izout, cell_id_out, idx_sorted, cell_id_indices = results

    assert np.allclose(x[idx_sorted], xout)
    assert np.allclose(y[idx_sorted], yout)
    assert np.allclose(z[idx_sorted], zout)


def test2():
    n1 = 123
    n2 = 4567
    Lbox = 1000.

    nx1, ny1, nz1 = 5, 6, 7
    lx, ly, lz = Lbox/nx1, Lbox/ny1, Lbox/nz1
    rmax = np.max((lx, ly, lz))
    nx2 = int((Lbox + 2*rmax)/lx)
    ny2 = int((Lbox + 2*rmax)/ly)
    nz2 = int((Lbox + 2*rmax)/lz)

    nbins = 6
    rbins = np.logspace(-1, np.log10(rmax), nbins)
    rbins_squared = rbins**2
    counts_mesh = np.zeros(nbins).astype(int)
    counts_nomesh = np.zeros(nbins).astype(int)

    x1, y1, z1 = random_points(n1, Lbox)

    results = calculate_chaining_mesh(x1, y1, z1, Lbox, Lbox, Lbox, nx1, ny1, nz1)
    x1out, y1out, z1out, ixout, iyout, izout, cell_id_out, idx_sorted, cell_id_indices = results

    x2_orig, y2_orig, z2_orig = random_points(n2, Lbox)

    for i in range(n1):
        px = x1[i]
        py = y1[i]
        pz = z1[i]
        for j in range(n2):
            qx = x2_orig[j]
            qy = y2_orig[j]
            qz = z2_orig[j]
            dx = px-qx
            dy = py-qy
            dz = pz-qz
            dsq = dx*dx + dy*dy + dz*dz

            k = nbins-1
            while dsq <= rbins_squared[k]:
                counts_nomesh[k-1] += 1
                k=k-1
                if k<=0:
                    break

    xyz_mins = (0, 0, 0)
    xyz_maxs = (Lbox, Lbox, Lbox)
    rmax_xyz = (rmax, rmax, rmax)
    period_xyz = (Lbox, Lbox, Lbox)

    _x2, _y2, _z2, __, __ = points_in_buffered_rectangle(
        x2_orig, y2_orig, z2_orig, xyz_mins, xyz_maxs, rmax_xyz, period_xyz)
    _x2 = _x2 + rmax
    _y2 = _y2 + rmax
    _z2 = _z2 + rmax

    results = calculate_chaining_mesh(
        _x2, _y2, _z2, Lbox+2*rmax, Lbox+2*rmax, Lbox+2*rmax, nx2, ny2, nz2)
    _x2out, _y2out, _z2out, ixout2, iyout2, izout2, cell_id_out2, idx_sorted2, cell_id_indices2 = results
    x2 = _x2out - rmax
    y2 = _y2out - rmax
    z2 = _z2out - rmax

    assert np.all(x2 >= -rmax)
    assert np.all(x2 <= Lbox+rmax)
    assert np.all(y2 >= -rmax)
    assert np.all(y2 <= Lbox+rmax)
    assert np.all(z2 >= -rmax)
    assert np.all(z2 <= Lbox+rmax)

    ncells1 = nx1*ny1*nz1
    for icell1 in range(ncells1):
        ifirst1 = cell_id_indices[icell1]
        ilast1 = cell_id_indices[icell1+1]

        x_icell1 = x1out[ifirst1:ilast1]
        y_icell1 = y1out[ifirst1:ilast1]
        z_icell1 = z1out[ifirst1:ilast1]

        Ni = ilast1 - ifirst1
        if Ni > 0:
            ix1 = icell1 // (ny1*nz1)
            iy1 = (icell1 - ix1*ny1*nz1) // nz1
            iz1 = icell1 - (ix1*ny1*nz1) - (iy1*nz1)

            leftmost_ix2 = ix1
            leftmost_iy2 = iy1
            leftmost_iz2 = iz1

            rightmost_ix2 = ix1 + 3
            rightmost_iy2 = iy1 + 3
            rightmost_iz2 = iz1 + 3

        for icell2_ix in range(leftmost_ix2, rightmost_ix2):
            for icell2_iy in range(leftmost_iy2, rightmost_iy2):
                for icell2_iz in range(leftmost_iz2, rightmost_iz2):

                    icell2 = icell2_ix*(ny2*nz2) + icell2_iy*nz2 + icell2_iz
                    ifirst2 = cell_id_indices2[icell2]
                    ilast2 = cell_id_indices2[icell2+1]

                    x_icell2 = x2[ifirst2:ilast2]
                    y_icell2 = y2[ifirst2:ilast2]
                    z_icell2 = z2[ifirst2:ilast2]

                    Nj = ilast2 - ifirst2
                    if Nj > 0:
                        for i in range(0, Ni):
                            x1tmp = x_icell1[i]
                            y1tmp = y_icell1[i]
                            z1tmp = z_icell1[i]
                            for j in range(0, Nj):
                                #calculate the square distance
                                dx = x1tmp - x_icell2[j]
                                dy = y1tmp - y_icell2[j]
                                dz = z1tmp - z_icell2[j]
                                dsq = dx*dx + dy*dy + dz*dz

                                k = nbins-1
                                while dsq <= rbins_squared[k]:
                                    counts_mesh[k] += 1
                                    k=k-1
                                    if k<0: break

    assert np.all(counts_mesh == counts_nomesh)




