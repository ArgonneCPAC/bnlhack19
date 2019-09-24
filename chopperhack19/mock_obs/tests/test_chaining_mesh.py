"""
"""
import numpy as np
from .chaining_mesh import calculate_chaining_mesh

DEFAULT_SEED = 43


def random_points(n, Lbox, seed=DEFAULT_SEED):
    """
    """
    rng = np.random.RandomState(seed)
    data = rng.uniform(0, 1, n*4)
    x, y, z = data[:n]*Lbox, data[n:2*n]*Lbox, data[2*n:3*n]*Lbox
    return x, y, z


def test1():
    """Verify that idx_sorted does indeed sort the output points by cell_id
    """
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
    """Compare brute-force pair counts to chaining-mesh pair counts
    """
    n1 = 123
    n2 = 4567
    Lbox = 1000.

    nx, ny, nz = 5, 5, 5
    lx, ly, lz = Lbox/nx, Lbox/ny, Lbox/nz
    rmax = np.max((lx, ly, lz))

    nbins = 6
    rbins = np.logspace(-1, np.log10(rmax), nbins)
    rbins_squared = rbins**2
    counts_mesh = np.zeros(nbins).astype(int)
    counts_nomesh = np.zeros(nbins).astype(int)

    x1, y1, z1 = random_points(n1, Lbox, 0)
    results = calculate_chaining_mesh(x1, y1, z1, Lbox, Lbox, Lbox, nx, ny, nz)
    x1out, y1out, z1out, ixout, iyout, izout, cell_id_out, idx_sorted, cell_id_indices = results

    x2, y2, z2 = random_points(n2, Lbox, 1)
    results2 = calculate_chaining_mesh(x2, y2, z2, Lbox, Lbox, Lbox, nx, ny, nz)
    x2out, y2out, z2out, ix2out, iy2out, iz2out, cell_id2_out, idx_sorted2, cell_id2_indices = results2

    for i in range(n1):
        px = x1[i]
        py = y1[i]
        pz = z1[i]
        for j in range(n2):
            qx = x2[j]
            qy = y2[j]
            qz = z2[j]
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

    ncells = nx*ny*nz
    for icell1 in range(ncells):
        ifirst1 = cell_id_indices[icell1]
        ilast1 = cell_id_indices[icell1+1]

        x_icell1 = x1out[ifirst1:ilast1]
        y_icell1 = y1out[ifirst1:ilast1]
        z_icell1 = z1out[ifirst1:ilast1]

        Ni = ilast1 - ifirst1
        if Ni > 0:
            ix1 = icell1 // (ny*nz)
            iy1 = (icell1 - ix1*ny*nz) // nz
            iz1 = icell1 - (ix1*ny*nz) - (iy1*nz)

            leftmost_ix2 = max(0, ix1-1)
            leftmost_iy2 = max(0, iy1-1)
            leftmost_iz2 = max(0, iz1-1)

            rightmost_ix2 = min(ix1+2, nx)
            rightmost_iy2 = min(iy1+2, ny)
            rightmost_iz2 = min(iz1+2, nz)

            for icell2_ix in range(leftmost_ix2, rightmost_ix2):
                for icell2_iy in range(leftmost_iy2, rightmost_iy2):
                    for icell2_iz in range(leftmost_iz2, rightmost_iz2):

                        icell2 = icell2_ix*(ny*nz) + icell2_iy*nz + icell2_iz
                        ifirst2 = cell_id2_indices[icell2]
                        ilast2 = cell_id2_indices[icell2+1]

                        x_icell2 = x2out[ifirst2:ilast2]
                        y_icell2 = y2out[ifirst2:ilast2]
                        z_icell2 = z2out[ifirst2:ilast2]

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
                                        counts_mesh[k-1] += 1
                                        k=k-1
                                        if k<0: break

    print(counts_nomesh)
    print(counts_mesh)
    assert np.all(counts_mesh == counts_nomesh)

