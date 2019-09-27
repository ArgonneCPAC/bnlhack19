import math
from numba import cuda


def combined_kernel(
        m1, mjac1, m1sigma, x1, y1, z1,
        m2, mjac2, m2sigma, x2, y2, z2,
        mbins, r2bins, mbins_for_r2,
        mhist, mhist_jac, r2hist, r2hist_jac):

    start = cuda.grid(1)
    stride = cuda.gridsize(1)

    n_1 = x1.shape[0]
    n_2 = x2.shape[0]

    n_params = mjac1.shape[1]

    n_mbins = mbins.shape[0]
    n_mbins_for_r2 = mbins_for_r2.shape[0]
    n_r2bins = r2bins.shape[0]-1
    dlogr = math.log(r2bins[1] / r2bins[0]) / 2
    logminr = math.log(r2bins[0]) / 2

    root2 = math.sqrt(2)
    rootpi = math.sqrt(math.pi)

    for i in range(start, n_1, stride):
        sigma1_sqrt2 = m1sigma[i] * root2
        sigma1_sqrt2pi = sigma1_sqrt2 * rootpi

        z1 = (m1[i] - mbins[0]) / sigma1_sqrt2
        last_cdf = (1 + math.erf(z1)) / 2
        last_cdf_deriv = math.exp(-z1*z1) / sigma1_sqrt2pi

        for k in range(n_mbins):
            z1 = (m1[i] - mbins[k+1]) / sigma1_sqrt2
            new_cdf = (1 + math.erf(z1)) / 2
            new_cdf_deriv = math.exp(-z1*z1) / sigma1_sqrt2pi

            # get the hist weight
            cuda.atomic.add(mhist, k, last_cdf - new_cdf)

            # do the derivs
            dwgt = last_cdf_deriv - new_cdf_deriv
            for _p in range(n_params):
                cuda.atomic.add(
                    mhist_jac,
                    k * n_params + _p,
                    dwgt * mjac1[i, _p])

            last_cdf = new_cdf
            last_cdf_deriv = new_cdf_deriv

        for j in range(n_2):
            dx = x1[i] - x2[j]
            dy = y1[i] - y2[j]
            dz = z1[i] - z2[j]
            dsq = cuda.fma(dx, dx, cuda.fma(dy, dy, dz * dz))

            rbin = int((math.log(dsq)/2 - logminr) / dlogr)
            if rbin >= 0 and rbin < n_r2bins:
                sigma2_sqrt2 = m2sigma[i] * root2
                sigma2_sqrt2pi = sigma2_sqrt2 * rootpi

                z1 = (m1[i] - mbins_for_r2[0]) / sigma1_sqrt2
                last_cdf1 = (1 + math.erf(z1)) / 2
                last_cdf_deriv1 = math.exp(-z1*z1) / sigma1_sqrt2pi

                for k1 in range(n_mbins_for_r2):
                    z1 = (m1[i] - mbins_for_r2[k1+1]) / sigma1_sqrt2
                    new_cdf1 = (1 + math.erf(z1)) / 2
                    new_cdf_deriv1 = math.exp(-z1*z1) / sigma1_sqrt2pi

                    w1 = last_cdf1 - new_cdf1
                    dw1 = last_cdf_deriv1 - new_cdf_deriv1

                    # point 2
                    z2 = (m2[j] - mbins_for_r2[k1]) / sigma2_sqrt2
                    last_cdf2 = (1 + math.erf(z2)) / 2
                    last_cdf_deriv2 = math.exp(-z2*z2) / sigma2_sqrt2pi

                    for k2 in range(k1, n_mbins_for_r2):
                        z2 = (m2[i] - mbins_for_r2[k2+1]) / sigma2_sqrt2
                        new_cdf2 = (1 + math.erf(z2)) / 2
                        new_cdf_deriv2 = math.exp(-z2*z2) / sigma2_sqrt2pi

                        w2 = last_cdf2 - new_cdf2
                        dw2 = last_cdf_deriv2 - new_cdf_deriv2

                        cuda.atomic.add(
                            r2hist,
                            k1 * n_mbins_for_r2 + k2,
                            w1 * w2)

                        for _p in range(n_params):
                            cuda.atomic.add(
                                r2hist_jac,
                                (k1 * n_mbins_for_r2 + k2) * n_params + _p,
                                (dw1 * mjac1[i, _p] * w2 +
                                 w1 * dw2 * mjac2[j, _p]))

                        last_cdf2 = new_cdf2
                        last_cdf_deriv2 = new_cdf_deriv2

                    last_cdf1 = new_cdf1
                    last_cdf_deriv1 = new_cdf_deriv1
