"""
"""
import numpy as np
from .generate_test_data import random_weighted_points, DEFAULT_RBINS_SQUARED
from .gaussian_weighted_pair_counts import count_weighted_pairs_3d_cpu, count_weighted_pairs_3d_cuda


def test_correct_counts():
    n1, n2 = 5000, 3333
    Lbox = 888.

    cpu_result = np.zeros_like(DEFAULT_RBINS_SQUARED)
    cuda_result = np.zeros_like(DEFAULT_RBINS_SQUARED)
    x1, y1, z1, w1 = random_weighted_points(n1, Lbox, 0)
    x2, y2, z2, w2 = random_weighted_points(n2, Lbox, 1)
    count_weighted_pairs_3d_cpu(x1, y1, z1, w1, x2, y2, z2, w2, DEFAULT_RBINS_SQUARED, cpu_result)
    count_weighted_pairs_3d_cuda(x1, y1, z1, w1, x2, y2, z2, w2, DEFAULT_RBINS_SQUARED, cuda_result)
    assert np.allclose(cpu_result, cuda_result), "Incorrect pair counts with cuda"
