import sys
import cupy as cp
import numpy as np
from numba import cuda

# parameters and test data
blocks = 512
threads = 512
if len(sys.argv) > 1:
    npoints = int(sys.argv[1])
else:
    npoints = 100_000

rng = np.random.RandomState(seed=42)
x1 = rng.uniform(size=npoints).astype(np.float32)
x2 = rng.uniform(size=npoints).astype(np.float32)

# cupy
CUDA_SRC = """
extern "C"{

__global__ void sum_arrays_cp(float* x1, float* x2, float* result, int n1) {
    size_t start = threadIdx.x + blockIdx.x * blockDim.x;
    size_t stride = blockDim.x * gridDim.x;

    for (size_t i = start; i < n1; i += stride) {
        result[i] = x1[i] + x2[i];
    }
}

}
"""

# compile and load CUDA kernel using CuPy
sum_arrays_cp = cp.RawKernel(CUDA_SRC, 'sum_arrays_cp')

# array init
result = np.zeros_like(x1).astype(np.float32)

d_x1 = cp.asarray(x1, dtype=cp.float32)
d_x2 = cp.asarray(x2, dtype=cp.float32)
d_result = cp.asarray(result, dtype=cp.float32)

# for GPU timing using CuPy
start = cp.cuda.Event()
end = cp.cuda.Event()
timing_cp = 0

# running the kernel using CuPy's functionality
for i in range(4):
    if i > 0:  # warm-up not needed if using RawModule
        start.record()
    sum_arrays_cp(
        (blocks,), (threads,), (d_x1, d_x2, d_result, cp.int32(d_x1.shape[0])))
    if i > 0:  # warm-up not needed if using RawModule
        end.record()
        end.synchronize()
        timing_cp += cp.cuda.get_elapsed_time(start, end)

print('launching CUDA kernel from CuPy took', timing_cp/3, 'ms in average')
d_result_cp = d_result.copy()

# for GPU timing using Numba
start = cuda.event()
end = cuda.event()
timing_nb = 0

d_x1 = cuda.to_device(x1.astype(np.float32))
d_x2 = cuda.to_device(x2.astype(np.float32))
d_result = cuda.device_array_like(result.astype(np.float32))


# running the Numba jit kernel
@cuda.jit
def sum_arrays_nb(x1, x2, result):
    start = cuda.grid(1)
    stride = cuda.gridsize(1)

    n1 = x1.shape[0]

    for i in range(start, n1, stride):
        result[i] = x1[i] + x2[i]


# this works because CuPy arrays have the __cuda_array_interface__ attribute,
# which is accepted by Numba kernels, so you don't have to create the arrays
# again using Numba's API
for i in range(4):
    if i > 0:
        start.record()
    sum_arrays_nb[blocks, threads](d_x1, d_x2, d_result)
    if i > 0:
        end.record()
        end.synchronize()
        timing_nb += cuda.event_elapsed_time(start, end)

print('launching Numba jit kernel took', timing_nb/3, 'ms in average')
d_result_nb = d_result.copy_to_host()

# check that the CUDA kernel agrees with the Numba kernel
assert cp.allclose(d_result_cp, d_result_nb, rtol=5E-4)


# print(sum_arrays_nb.inspect_types())
