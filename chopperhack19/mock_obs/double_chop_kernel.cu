extern "C"{

__global__ void double_chop_pairs_pure_cuda(
    float* x1, float* y1, float* z1, float* w1, int* cell1,
    float* x2, float* y2, float* z2, float* w2, int* indx2,
    float* rbins_squared, float* result,
    int n1, int nbins) {// array attributes must be explicitly passed in.
    /* 
       Direct translation of the Numba "double_chop_pairs_cuda" kernel.
    */

    size_t start = threadIdx.x + blockIdx.x * blockDim.x;
    size_t stride = blockDim.x * gridDim.x;

//    if (start == 0)
//        printf("%i, %i\n", n1, nbins);

    for (size_t i = start; i < n1; i += stride) {
        float px = x1[i];
        float py = y1[i];
        float pz = z1[i];
        float pw = w1[i];

        int cell1_i = cell1[i];
        int first = indx2[cell1_i];
        int last = indx2[cell1_i+1];

        for (size_t j = first; j < last; j++) {
            float qx = x2[j];
            float qy = y2[j];
            float qz = z2[j];
            float qw = w2[j];

            float dx = px - qx;
            float dy = py - qy;
            float dz = pz - qz;
            float wprod = pw * qw;
            float dsq = dx * dx + dy * dy + dz * dz;

            size_t k = nbins - 1;
            while (dsq <= rbins_squared[k]) {
                atomicAdd(&(result[k-1]), wprod);
                k -= 1;
                if (k <= 0) break;
            }
        }
    }
}

}
