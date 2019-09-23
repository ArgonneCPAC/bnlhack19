"""
"""
import numpy as np
from chaining_mesh import calculate_chaining_mesh


DEFAULT_SEED = 43


def random_points(n, Lbox, seed=DEFAULT_SEED):
    """
    """
    rng = np.random.RandomState(seed)
    data = rng.uniform(0, 1, n*4)
    x, y, z = data[:n]*Lbox, data[n:2*n]*Lbox, data[2*n:3*n]*Lbox
    return x, y, z


def test1():
    n = 12345
    Lbox = 1000.
    x, y, z = random_points(n, Lbox)

    nx, ny, nz = 3, 4, 5

    results = calculate_chaining_mesh(x, y, z, Lbox, Lbox, Lbox, nx, ny, nz)
    xout, yout, zout, ixout, iyout, izout, cell_id_out, idx_sorted, cell_id_indices = results

    assert np.allclose(x[idx_sorted], xout)
    assert np.allclose(y[idx_sorted], yout)
    assert np.allclose(z[idx_sorted], zout)


