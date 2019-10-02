import sys
import time

import cupy as cp
import numpy as np
from numba import cuda

if len(sys.argv) > 2:
    kind = sys.argv[2]
else:
    kind = 'both'

# parameters
blocks = 512
threads = 512
if len(sys.argv) > 1:
    npoints = int(sys.argv[1])
else:
    npoints = 100_000
n1 = npoints
n2 = npoints

# make the data
DEFAULT_RBINS_SQUARED = (np.logspace(
    np.log10(0.1/1e3), np.log10(40/1e3), 20)**2).astype(np.float32)
rng = np.random.RandomState(seed=42)
x1, y1, z1, w1 = rng.uniform(size=(4, n1)).astype(np.float32)
x2, y2, z2, w2 = rng.uniform(size=(4, n1)).astype(np.float32)

# array init
result = np.zeros_like(DEFAULT_RBINS_SQUARED)[:-1].astype(np.float32)

if kind in ['both', 'cupy']:

    source_code = """\
    extern "C"{

    __global__ void brute_force_pairs_kernel(
        float* x1, float* y1, float* z1, float* w1,
        float* x2, float* y2, float* z2, float* w2,
        float* rbins_squared, float* result,
        int n1, int n2, int nbins) {

        size_t start = threadIdx.x + blockIdx.x * blockDim.x;
        size_t stride = blockDim.x * gridDim.x;
        float g = 0;

        for (size_t i = start; i < n1; i += stride) {
            float px = x1[i];
            float py = y1[i];
            float pz = z1[i];
            float pw = w1[i];

            for (size_t j = 0; j < n2; j++) {
                float qx = x2[j];
                float qy = y2[j];
                float qz = z2[j];
                float qw = w2[j];

                float dx = px - qx;
                float dy = py - qy;
                float dz = pz - qz;
                float wprod = pw * qw;
                float dsq = dx * dx + dy * dy + dz * dz;

                g += (dsq * wprod);
            }
        }

        result[0] = g;
    }

    }
    """

    # compile and load CUDA kernel using CuPy
    brute_force_pairs_kernel = cp.RawKernel(
        source_code, 'brute_force_pairs_kernel')

    d_x1 = cp.asarray(x1, dtype=cp.float32)
    d_y1 = cp.asarray(y1, dtype=cp.float32)
    d_z1 = cp.asarray(z1, dtype=cp.float32)
    d_w1 = cp.asarray(w1, dtype=cp.float32)

    d_x2 = cp.asarray(x2, dtype=cp.float32)
    d_y2 = cp.asarray(y2, dtype=cp.float32)
    d_z2 = cp.asarray(z2, dtype=cp.float32)
    d_w2 = cp.asarray(w2, dtype=cp.float32)

    d_rbins_squared = cp.asarray(DEFAULT_RBINS_SQUARED, dtype=cp.float32)
    d_result_cp = cp.asarray(result, dtype=cp.float32)

    # for GPU timing using CuPy
    start = cp.cuda.Event()
    end = cp.cuda.Event()
    timing_cp = 0
    timing_cp_wall = 0

    # running the kernel using CuPy's functionality
    for i in range(4):
        if i > 0:  # warm-up not needed if using RawModule
            start.record()
            _s = time.time()
        brute_force_pairs_kernel(
            (blocks,), (threads,),
            (d_x1, d_y1, d_z1, d_w1,
             d_x2, d_y2, d_z2, d_w2,
             d_rbins_squared, d_result_cp,
             cp.int32(d_x1.shape[0]),
             cp.int32(d_x2.shape[0]),
             cp.int32(d_rbins_squared.shape[0]))
        )
        if i > 0:  # warm-up not needed if using RawModule
            end.record()
            end.synchronize()
            _e = time.time()
            timing_cp += cp.cuda.get_elapsed_time(start, end)
            timing_cp_wall += (_e - _s)

    print('cupy+CUDA events:', timing_cp/3, 'ms')
    print('cupy+CUDA wall  :', timing_cp_wall/3*1000, 'ms')
    d_result_cp = d_result_cp.copy()


if kind in ['both', 'numba']:
    # for GPU timing using Numba
    @cuda.jit
    def count_weighted_pairs_3d_cuda(
            x1, y1, z1, w1, x2, y2, z2, w2, rbins_squared, result):
        start = cuda.grid(1)
        stride = cuda.gridsize(1)

        n1 = x1.shape[0]
        n2 = x2.shape[0]
        g = 0

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

                g += (dsq * wprod)

        result[0] = g

    start = cuda.event()
    end = cuda.event()
    timing_nb = 0
    timing_nb_wall = 0

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
    for i in range(4):
        if i > 0:
            start.record()
            _s = time.time()
        count_weighted_pairs_3d_cuda[blocks, threads](
            d_x1, d_y1, d_z1, d_w1,
            d_x2, d_y2, d_z2, d_w2,
            d_rbins_squared, d_result_nb)
        if i > 0:
            end.record()
            end.synchronize()
            _e = time.time()
            timing_nb += cuda.event_elapsed_time(start, end)
            timing_nb_wall += (_e - _s)

    print('numba events:', timing_nb/3, 'ms')
    print('numba wall  :', timing_nb_wall/3*1000, 'ms')

    # print(count_weighted_pairs_3d_cuda.inspect_types())

    print("numba ptx:", count_weighted_pairs_3d_cuda.ptx)

if kind in ['both'] and npoints <= 10:
    # check that the CUDA kernel agrees with the Numba kernel
    assert cp.allclose(d_result_cp, d_result_nb, rtol=5E-4)
