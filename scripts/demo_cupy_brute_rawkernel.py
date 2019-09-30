import os
import time
import sys

import cupy as cp
import numpy as np
from numba import cuda

from chopperhack19.mock_obs.tests import random_weighted_points
from chopperhack19.mock_obs.tests.generate_test_data import (
    DEFAULT_RBINS_SQUARED)
from chopperhack19.mock_obs import count_weighted_pairs_3d_cuda

########################################################################
# This demo shows how to compile a CUDA .cu file, load a particular CUDA
# kernel, launch it with CuPy arrays. Also shows how to make CuPy and
# Numba work together
########################################################################


filepath = os.path.abspath(os.path.join(
    __file__, "../../chopperhack19/mock_obs/brute_force_pairs_kernel.cu"))
with open(filepath) as f:
    source_code = f.read()

# compile and load CUDA kernel using CuPy
# before CuPy v7.0.0b3:
# in this case, compilation happens at the first invocation of the kernel,
# not the declaration time
brute_force_pairs_kernel = cp.RawKernel(
    source_code, 'brute_force_pairs_kernel')

# starting CuPy v7.0.0b3:
# RawModule is suitable for importing a large CUDA codebase
# the compilation happens when initializing the RawModule instance
# mod = cp.RawModule(source_code)
# double_chop_kernel = mod.get_function('double_chop_pairs_pure_cuda')

# parameters
blocks = 512
threads = 512
if len(sys.argv) > 1:
    npoints = int(sys.argv[1])
else:
    npoints = 100_000

Lbox = 1000.

# array init
# CuPy functionalities should be used to avoid unnecessary computation
# and transfer, which I didn't do here as it's midnight...
result = np.zeros_like(DEFAULT_RBINS_SQUARED)[:-1].astype(cp.float32)

n1 = npoints
n2 = npoints
x1, y1, z1, w1 = random_weighted_points(n1, Lbox, 0)
x2, y2, z2, w2 = random_weighted_points(n2, Lbox, 1)

d_x1 = cp.asarray(x1, dtype=cp.float32)
d_y1 = cp.asarray(y1, dtype=cp.float32)
d_z1 = cp.asarray(z1, dtype=cp.float32)
d_w1 = cp.asarray(w1, dtype=cp.float32)

d_x2 = cp.asarray(x2, dtype=cp.float32)
d_y2 = cp.asarray(y2, dtype=cp.float32)
d_z2 = cp.asarray(z2, dtype=cp.float32)
d_w2 = cp.asarray(w2, dtype=cp.float32)

d_rbins_squared = cp.asarray(DEFAULT_RBINS_SQUARED, dtype=cp.float32)
d_result = cp.asarray(result, dtype=cp.float32)
d_result_nb = cp.asarray(result, dtype=cp.float32)

# for GPU timing using CuPy
start = cp.cuda.Event()
end = cp.cuda.Event()
timing_cp = 0

# running the kernel using CuPy's functionality
for i in range(4):
    if i > 0:  # warm-up not needed if using RawModule
        start.record()
    if i == 1:
        _start = time.time()
    brute_force_pairs_kernel(
        (blocks,), (threads,),
        (d_x1, d_y1, d_z1, d_w1,
         d_x2, d_y2, d_z2, d_w2,
         d_rbins_squared, d_result,
         cp.int32(d_x1.shape[0]),
         cp.int32(d_x1.shape[0]),
         cp.int32(d_rbins_squared.shape[0]))
    )
    if i > 0:  # warm-up not needed if using RawModule
        end.record()
        end.synchronize()
        timing_cp += cp.cuda.get_elapsed_time(start, end)

_end = time.time()

print('launching CUDA kernel from CuPy took', timing_cp/3, 'ms in average')
print('wall time:', (_end - _start)/3)
d_result_cp = d_result.copy()

# for GPU timing using Numba
start = cuda.event()
end = cuda.event()
timing_nb = 0

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
d_result_nb = cuda.device_array_like(result.astype(np.float32))


# running the Numba jit kernel
# this works because CuPy arrays have the __cuda_array_interface__ attribute,
# which is accepted by Numba kernels, so you don't have to create the arrays
# again using Numba's API
for i in range(4):
    if i > 0:
        start.record()
    if i == 1:
        _start = time.time()
    count_weighted_pairs_3d_cuda[blocks, threads](
        d_x1, d_y1, d_z1, d_w1,
        d_x2, d_y2, d_z2, d_w2,
        d_rbins_squared, d_result_nb)
    if i > 0:
        end.record()
        end.synchronize()
        timing_nb += cuda.event_elapsed_time(start, end)

d_result_nb = d_result_nb.copy_to_host()
_end = time.time()

print('launching Numba jit kernel took', timing_nb/3, 'ms in average')
print('wall time:', (_end - _start)/3)

# check that the CUDA kernel agrees with the Numba kernel
assert cp.allclose(d_result_cp, d_result_nb, rtol=5E-4)
