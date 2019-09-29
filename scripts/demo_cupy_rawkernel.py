import os, sys

import cupy as cp
import numpy as np
from numba import cuda

from chopperhack19.mock_obs.tests import random_weighted_points
from chopperhack19.mock_obs.tests.generate_test_data import (
    DEFAULT_RBINS_SQUARED)
from chopperhack19.mock_obs import chaining_mesh as cm
from chopperhack19.mock_obs.double_chop_kernel import double_chop_pairs_cuda

########################################################################
# This demo shows how to compile a CUDA .cu file, load a particular CUDA
# kernel, launch it with CuPy arrays. Also shows how to make CuPy and 
########################################################################


filepath = os.path.abspath(os.path.join(__file__, "../../chopperhack19/mock_obs/double_chop_kernel.cu"))
with open(filepath) as f:
    source_code = f.read()

# compile and load CUDA kernel using CuPy
# before CuPy v7.0.0b3:
# in this case, compilation happens at the first invocation of the kernel, 
# not the declaration time
double_chop_kernel = cp.RawKernel(source_code, 'double_chop_pairs_pure_cuda')

## starting CuPy v7.0.0b3:
## RawModule is suitable for importing a large CUDA codebase
## the compilation happens when initializing the RawModule instance
#mod = cp.RawModule(source_code)
#double_chop_kernel = mod.get_function('double_chop_pairs_pure_cuda')

# parameters
blocks = 512
threads = 512
npoints = 200013
nmesh1 = 4
nmesh2 = 16

Lbox = 1000.

# array init
# CuPy functionalities should be used to avoid unnecessary computation
# and transfer, which I didn't do here as it's midnight...
result = np.zeros_like(DEFAULT_RBINS_SQUARED)[:-1].astype(cp.float32)

n1 = npoints
n2 = npoints
x1, y1, z1, w1 = random_weighted_points(n1, Lbox, 0)
x2, y2, z2, w2 = random_weighted_points(n2, Lbox, 1)

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

d_x1 = cp.asarray(x1out, dtype=cp.float32)
d_y1 = cp.asarray(y1out, dtype=cp.float32)
d_z1 = cp.asarray(z1out, dtype=cp.float32)
d_w1 = cp.asarray(w1out, dtype=cp.float32)
d_cell1out = cp.asarray(cell1out, dtype=cp.int32)

d_x2 = cp.asarray(x2out, dtype=cp.float32)
d_y2 = cp.asarray(y2out, dtype=cp.float32)
d_z2 = cp.asarray(z2out, dtype=cp.float32)
d_w2 = cp.asarray(w2out, dtype=cp.float32)
d_indx2 = cp.asarray(indx2, dtype=cp.int32)

d_rbins_squared = cp.asarray(DEFAULT_RBINS_SQUARED, dtype=cp.float32)
d_result = cp.asarray(result, dtype=cp.float32)

# for GPU timing using CuPy
start = cp.cuda.Event()
end = cp.cuda.Event()
timing_cp = 0

# running the kernel using CuPy's functionality
for i in range(4):
    d_result[...] = 0.
    if i > 0:  # warm-up not needed if using RawModule
        start.record()
    double_chop_kernel((blocks,), (threads,),
                       (d_x1, d_y1, d_z1, d_w1, d_cell1out,
                        d_x2, d_y2, d_z2, d_w2, d_indx2,
                        d_rbins_squared, d_result,
                        cp.int32(d_x1.shape[0]), cp.int32(d_rbins_squared.shape[0]))
                      )
    if i > 0:  # warm-up not needed if using RawModule
        end.record()
        end.synchronize()
        timing_cp += cp.cuda.get_elapsed_time(start, end)
#cp.cuda.Stream.null.synchronize()
print('launching CUDA kernel from CuPy took', timing_cp/3, 'ms in average')
d_result_cp = d_result.copy()

# for GPU timing using Numba
start = cuda.event()
end = cuda.event()
timing_nb = 0

# running the Numba jit kernel
# this works because CuPy arrays have the __cuda_array_interface__ attribute,
# which is accepted by Numba kernels, so you don't have to create the arrays
# again using Numba's API
for i in range(4):
    d_result[...] = 0.
    if i > 0:
        start.record()
    double_chop_pairs_cuda[blocks, threads](d_x1, d_y1, d_z1, d_w1, d_cell1out,
                                            d_x2, d_y2, d_z2, d_w2, d_indx2,
                                            d_rbins_squared, d_result)
    if i > 0:
        end.record()
        end.synchronize()
        timing_nb += cuda.event_elapsed_time(start, end)
print('launching Numba jit kernel took', timing_nb/3, 'ms in average')
d_result_nb = d_result.copy()

# check that the CUDA kernel agrees with the Numba kernel
assert cp.allclose(d_result_cp, d_result_nb, rtol=5E-4)
