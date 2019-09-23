"""Simple script demonstrating a timing test of the cuda implementation
of the brute-force pair-counter
"""
from numba import cuda
import numpy as np
from chopperhack19 import count_weighted_pairs_3d
from time import time


#  Set the specs of the problem
npts1 = int(4096)*4
npts2 = int(4096)*64
Lbox = 250.
rmax = 30.
nrbins = 15

#  Define some dummy data on the CPU
rbins = np.logspace(-1, np.log10(rmax), nrbins)
rbins_squared = rbins**2

rng1 = np.random.RandomState(1)
pos1 = rng1.uniform(-rmax, Lbox+rmax, size=npts1*3)
x1, y1, z1 = pos1[0:npts1], pos1[npts1:2*npts1], pos1[2*npts1:]
w1 = rng1.uniform(0, 1, npts1)

rng2 = np.random.RandomState(2)
pos2 = rng1.uniform(-rmax, Lbox+rmax, size=npts2*3)
x2, y2, z2 = pos2[0:npts2], pos2[npts2:2*npts2], pos2[2*npts2:]
w2 = rng2.uniform(0, 1, npts2)

results = np.zeros_like(rbins_squared)


#  Transfer data to the GPU
d_x1 = cuda.to_device(x1)
d_y1 = cuda.to_device(y1)
d_z1 = cuda.to_device(z1)
d_w1 = cuda.to_device(w1)

d_x2 = cuda.to_device(x2)
d_y2 = cuda.to_device(y2)
d_z2 = cuda.to_device(z2)
d_w2 = cuda.to_device(w2)

d_rbins_squared = cuda.to_device(rbins_squared)

d_results = cuda.device_array_like(results)

threads_per_warp = 32
warps_per_block = 4
threads_per_block = warps_per_block*threads_per_warp
blocks_per_grid = 32


start = time()
count_weighted_pairs_3d[blocks_per_grid, threads_per_block](
    d_x1, d_y1, d_z1, d_w1, d_x2, d_y2, d_z2, d_w2, d_rbins_squared, d_results)

results_host = d_results.copy_to_host()
end = time()
print(results_host)

msg = "\nFor (n1, n2) = ({0}, {1}), pair-counting runtime = {2:2f} seconds"
print(msg.format(npts1, npts2, end-start))
